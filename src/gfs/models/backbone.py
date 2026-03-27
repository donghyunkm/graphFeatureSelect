"""GNN backbone for spatial transcriptomics graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

# Try torch_scatter first, fall back to PyG's scatter
try:
    from torch_scatter import scatter_mean
except ImportError:
    from torch_geometric.utils import scatter

    def scatter_mean(src, index, dim=0):
        return scatter(src, index, dim=dim, reduce="mean")


def _build_gnn_layer(gnn_type: str, in_ch: int, out_ch: int, heads: int = 1) -> nn.Module:
    """Factory for GNN layer types."""
    if gnn_type == "gat":
        return GATv2Conv(in_ch, out_ch, heads=heads, concat=False, add_self_loops=False, bias=False)
    elif gnn_type == "sage":
        return SAGEConv(in_ch, out_ch)
    elif gnn_type == "gcn":
        return GCNConv(in_ch, out_ch, cached=False, normalize=True)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")


class GNNBackbone(nn.Module):
    """GNN backbone that takes masked gene expression + spatial coords and returns node embeddings.

    Architecture (from "Classic GNNs are strong baselines", Luo et al. 2024):
    - Optional pre-linear layer (gene_ch -> hid_ch)
    - Stack of GNN layers with optional: residual connections, normalization, JK skip
    - Optional spatial coordinate projection added to final embedding
    - Optional expression residual path (MLP from masked expression to output)

    Args:
        gene_ch: number of input gene features (after masking, still n_genes)
        spatial_ch: number of spatial coordinate dims (2 or 3)
        hid_ch: hidden dimension
        n_layers: number of GNN layers
        gnn_type: "gat", "sage", or "gcn"
        dropout: dropout probability
        heads: attention heads (for GAT only)
        pre_linear: if True, apply linear before first GNN layer
        residual: if True, add linear skip to each GNN layer
        layer_norm: if True, apply LayerNorm after each GNN layer
        batch_norm: if True, apply BatchNorm after each GNN layer
        jk: if True, sum all layer outputs (jumping knowledge)
        xyz_proj: if True, add linear projection of spatial coords to output
        x_residual: if True, add MLP residual from masked expression to output
    """

    def __init__(
        self,
        gene_ch: int,
        spatial_ch: int,
        hid_ch: int,
        n_layers: int = 2,
        gnn_type: str = "gat",
        dropout: float = 0.0,
        heads: int = 1,
        pre_linear: bool = True,
        residual: bool = True,
        layer_norm: bool = True,
        batch_norm: bool = False,
        jk: bool = False,
        xyz_proj: bool = False,
        x_residual: bool = False,
    ):
        super().__init__()
        self.hid_ch = hid_ch
        self.dropout = dropout
        self.pre_linear = pre_linear
        self.use_residual = residual
        self.use_ln = layer_norm
        self.use_bn = batch_norm
        self.use_jk = jk

        # Optional pre-linear
        if pre_linear:
            self.lin_in = nn.Linear(gene_ch, hid_ch)
            first_in = hid_ch
        else:
            first_in = gene_ch

        # GNN layers (unified — no special first layer)
        self.gnn_layers = nn.ModuleList()
        self.res_lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_ch = first_in if i == 0 else hid_ch
            self.gnn_layers.append(_build_gnn_layer(gnn_type, in_ch, hid_ch, heads))
            if residual:
                self.res_lins.append(nn.Linear(in_ch, hid_ch))
            if layer_norm:
                self.lns.append(nn.LayerNorm(hid_ch))
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(hid_ch))

        # Optional xyz projection (spatial_ch -> hid_ch)
        self.xyz_proj = nn.Linear(spatial_ch, hid_ch) if xyz_proj else None

        # Optional expression residual (gene_ch -> hid_ch via MLP)
        if x_residual:
            self.x_res_mlp = nn.Sequential(
                nn.Linear(gene_ch, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, hid_ch),
            )
        else:
            self.x_res_mlp = None

    def forward(
        self,
        gene_exp: torch.Tensor,
        edge_index: torch.Tensor,
        xyz: torch.Tensor,
        subgraph_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            gene_exp: (n_nodes, gene_ch) masked gene expression
            edge_index: (2, n_edges) graph connectivity
            xyz: (n_nodes, spatial_ch) spatial coordinates
            subgraph_id: (n_nodes,) subgraph assignment (for XYZ centering)

        Returns:
            embeddings: (n_nodes, hid_ch) node embeddings
        """
        # Center XYZ per subgraph using scatter ops
        if subgraph_id is not None:
            xyz = self._center_xyz(xyz, subgraph_id)

        # Save raw inputs for residual paths
        x_raw = gene_exp

        # Pre-linear
        if self.pre_linear:
            x = self.lin_in(gene_exp)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = gene_exp

        # GNN layers
        x_final = 0
        for i, gnn in enumerate(self.gnn_layers):
            if self.use_residual:
                x = gnn(x, edge_index) + self.res_lins[i](x)
            else:
                x = gnn(x, edge_index)
            if self.use_ln:
                x = self.lns[i](x)
            elif self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_jk:
                x_final = x_final + x
            else:
                x_final = x

        # Add residual paths
        if self.x_res_mlp is not None:
            x_final = x_final + self.x_res_mlp(x_raw)
        if self.xyz_proj is not None:
            x_final = x_final + self.xyz_proj(xyz)

        return x_final

    @staticmethod
    def _center_xyz(xyz: torch.Tensor, subgraph_id: torch.Tensor) -> torch.Tensor:
        """Center spatial coordinates per subgraph using scatter ops."""
        means = scatter_mean(xyz, subgraph_id, dim=0)
        return xyz - means[subgraph_id]
