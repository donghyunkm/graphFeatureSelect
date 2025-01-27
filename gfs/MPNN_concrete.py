# Architecture is adapted from Luo et al. 2024 (Classic GNNs are strong baselines)
# Added a concrete layer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class MPNN_conc(torch.nn.Module):  # noqa: N801
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_mask,
        local_layers=2,
        dropout=0.5,
        heads=8,
        pre_linear=True,
        res=True,
        ln=True,
        bn=False,
        jk=True,
        x_res=True,
        gnn="gcn",
        xyz_status=True,
    ):
        super(MPNN_conc, self).__init__()

        self.concrete = nn.Parameter(torch.randn(n_mask, in_channels))
        self.dropout = dropout

        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.x_res = x_res
        self.xyz_status = xyz_status
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)

        if not self.pre_linear:
            if gnn == "gat":
                self.local_convs.append(
                    GATConv(in_channels, hidden_channels, heads=heads, concat=False, add_self_loops=False, bias=False)
                )
            elif gnn == "sage":
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            local_layers = local_layers - 1

        for _ in range(local_layers):
            if gnn == "gat":
                self.local_convs.append(
                    GATConv(
                        hidden_channels, hidden_channels, heads=heads, concat=False, add_self_loops=False, bias=False
                    )
                )
            elif gnn == "sage":
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

        if self.x_res:
            self.res_lin = torch.nn.Linear(in_channels, out_channels)
        if self.xyz_status:
            self.xyz_lin = torch.nn.Linear(2, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()
        self.res_lin.reset_parameters()
        self.xyz_lin.reset_parameters()
        self.concrete.reset_parameters()

    def forward(self, x, edge_index, xyz, temp, hard_):
        mask = F.gumbel_softmax(self.concrete, tau=temp, hard=hard_)
        mask = torch.sum(mask, axis=0)
        mask = torch.clamp(mask, min=0, max=1)
        x = mask * x

        if self.x_res:
            x_to_add = self.res_lin(x)
        if self.xyz_status:
            xyz = self.xyz_lin(xyz)

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0

        for i, local_conv in enumerate(self.local_convs):
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
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

    def concrete_argmax(self):
        # return F.softmax(self.concrete, dim=1)
        return torch.argmax(self.concrete, dim=1)
