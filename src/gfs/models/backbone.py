import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

from gfs.models.components import MLP
from gfs.models.feature_selection import get_feature_selector


class GnnFs(nn.Module):
    """
    GNN with composable feature selection.

    This model features:
    1. a GNN backbone adapted from `Classic GNNs are strong baselines...` (Luo et al. 2024)
    2. a pluggable feature selection layer

    Args:
        gene_ch (int): number of input features
        spatial_ch (int): number of spatial features
        hid_ch (int): number of hidden features
        out_ch (int): number of output features
        n_select (int): number of features to select
        local_layers (int): gnn depth
        dropout (float): dropout rate
        heads (int): number of heads
        pre_linear (bool): if True, use pre-linear layer
        res (bool): if True, use residual connections
        ln (bool): if True, use layer normalization
        bn (bool): if True, use batch normalization
        jk (bool): if True, use skip connections
        x_res (bool): if True, use x residual connections
        gnn (str): "gat", "sage", or "gcn"
        xyz_status (bool): if True, use xyz status
        fs_method (str): feature selection method
        focal_loss (bool): if True, use focal loss
        sigma (float): STG sigma parameter
        lam (float): regularization weight for STG
    """

    def __init__(
        self,
        gene_ch,
        spatial_ch,
        hid_ch,
        out_ch,
        n_select,
        local_layers,
        dropout,
        heads,
        pre_linear,
        res,
        ln,
        bn,
        jk,
        x_res,
        gnn,
        xyz_status,
        fs_method,
        focal_loss,
        sigma=1.0,
        lam=0.1,
    ):
        super().__init__()
        self.fs_method = fs_method
        self.dropout = dropout
        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.x_res = x_res
        self.xyz_status = xyz_status
        self.focal_loss = focal_loss
        self.lam = lam

        self.feature_selector = get_feature_selector(
            fs_method,
            n_select=n_select,
            gene_ch=gene_ch,
            sigma=sigma,
        )

        self.gnn_layers = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lin_in = nn.Linear(gene_ch, hid_ch)

        if not self.pre_linear:
            if gnn == "gat":
                self.gnn_layers.append(
                    GATv2Conv(gene_ch, hid_ch, heads=heads, concat=False, add_self_loops=False, bias=False)
                )
            elif gnn == "sage":
                self.gnn_layers.append(SAGEConv(gene_ch, hid_ch))
            else:
                self.gnn_layers.append(GCNConv(gene_ch, hid_ch, cached=False, normalize=True))
            self.lins.append(nn.Linear(gene_ch, hid_ch))
            self.lns.append(nn.LayerNorm(hid_ch))
            self.bns.append(nn.BatchNorm1d(hid_ch))
            local_layers = local_layers - 1

        for _ in range(local_layers):
            if gnn == "gat":
                self.gnn_layers.append(
                    GATv2Conv(hid_ch, hid_ch, heads=heads, concat=False, add_self_loops=False, bias=False)
                )
            elif gnn == "sage":
                self.gnn_layers.append(SAGEConv(hid_ch, hid_ch))
            else:
                self.gnn_layers.append(GCNConv(hid_ch, hid_ch, cached=False, normalize=True))
            self.lins.append(nn.Linear(hid_ch, hid_ch))
            self.lns.append(nn.LayerNorm(hid_ch))
            self.bns.append(nn.BatchNorm1d(hid_ch))

        self.pred_local = nn.Linear(hid_ch, out_ch)

        if self.x_res:
            self.res_lin = MLP(gene_ch, out_ch, [128, 128], nn.ReLU())
        if self.xyz_status:
            self.xyz_lin = nn.Linear(spatial_ch, out_ch)

    def forward(self, x, edge_index, xyz, subgraph_id, tau, hard_):
        for id in subgraph_id.unique():
            xyz_mean = torch.mean(xyz[subgraph_id == id])
            xyz[subgraph_id == id] = xyz[subgraph_id == id] - xyz_mean

        x = self.feature_selector(x, tau=tau, subgraph_id=subgraph_id)

        if self.x_res:
            x_to_add = self.res_lin(x)
        if self.xyz_status:
            xyz = self.xyz_lin(xyz)

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0

        for i, layer in enumerate(self.gnn_layers):
            if self.res:
                x = layer(x, edge_index) + self.lins[i](x)
            else:
                x = layer(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x

        x = self.pred_local(x_final)

        if self.x_res:
            x = x + x_to_add

        if self.xyz_status:
            x = x + xyz

        return x
