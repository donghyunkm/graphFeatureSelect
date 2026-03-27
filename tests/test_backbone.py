"""Integration tests for GNNBackbone and task heads."""

import pytest
import torch

from gfs.models.backbone import GNNBackbone
from gfs.models.heads import ClassificationHead, ReconstructionHead

# ---------------------------------------------------------------------------
# Constants matching dev data dimensions
# ---------------------------------------------------------------------------
N_NODES = 200
N_GENES = 485
N_SPATIAL = 2
HID_CH = 32
N_CLASSES = 20
N_EDGES = 2000
N_SUBGRAPHS = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gene_exp():
    return torch.randn(N_NODES, N_GENES)


@pytest.fixture
def edge_index():
    src = torch.randint(0, N_NODES, (N_EDGES,))
    dst = torch.randint(0, N_NODES, (N_EDGES,))
    return torch.stack([src, dst])


@pytest.fixture
def xyz():
    return torch.randn(N_NODES, N_SPATIAL)


@pytest.fixture
def subgraph_id():
    return torch.repeat_interleave(
        torch.arange(N_SUBGRAPHS), N_NODES // N_SUBGRAPHS
    )


# ---------------------------------------------------------------------------
# Backbone tests
# ---------------------------------------------------------------------------
class TestGNNBackbone:
    def test_backbone_output_shape(self, gene_exp, edge_index, xyz, subgraph_id):
        backbone = GNNBackbone(
            gene_ch=N_GENES, spatial_ch=N_SPATIAL, hid_ch=HID_CH,
            n_layers=2, gnn_type="gat", dropout=0.0, heads=1,
            pre_linear=True, residual=True, layer_norm=True, batch_norm=False,
            jk=False, xyz_proj=False, x_residual=False,
        )
        out = backbone(gene_exp, edge_index, xyz, subgraph_id)
        assert out.shape == (N_NODES, HID_CH)

    @pytest.mark.parametrize("gnn_type", ["gat", "sage", "gcn"])
    def test_backbone_gnn_types(self, gnn_type, gene_exp, edge_index, xyz, subgraph_id):
        backbone = GNNBackbone(
            gene_ch=N_GENES, spatial_ch=N_SPATIAL, hid_ch=HID_CH,
            n_layers=2, gnn_type=gnn_type,
        )
        out = backbone(gene_exp, edge_index, xyz, subgraph_id)
        assert out.shape == (N_NODES, HID_CH)

    @pytest.mark.parametrize(
        "opts",
        [
            dict(pre_linear=False),
            dict(jk=True),
            dict(xyz_proj=True),
            dict(x_residual=True),
            dict(pre_linear=True, jk=True, xyz_proj=True, x_residual=True),
        ],
        ids=["no_prelinear", "jk", "xyz_proj", "x_residual", "all_options"],
    )
    def test_backbone_with_options(self, opts, gene_exp, edge_index, xyz, subgraph_id):
        backbone = GNNBackbone(
            gene_ch=N_GENES, spatial_ch=N_SPATIAL, hid_ch=HID_CH,
            n_layers=2, gnn_type="gat", **opts,
        )
        out = backbone(gene_exp, edge_index, xyz, subgraph_id)
        assert out.shape == (N_NODES, HID_CH)

    def test_backbone_no_subgraph_id(self, gene_exp, edge_index, xyz):
        backbone = GNNBackbone(
            gene_ch=N_GENES, spatial_ch=N_SPATIAL, hid_ch=HID_CH,
        )
        out = backbone(gene_exp, edge_index, xyz, subgraph_id=None)
        assert out.shape == (N_NODES, HID_CH)

    def test_xyz_centering(self, xyz, subgraph_id):
        centered = GNNBackbone._center_xyz(xyz, subgraph_id)
        for sg in range(N_SUBGRAPHS):
            sg_xyz = centered[subgraph_id == sg]
            assert sg_xyz.mean(dim=0).abs().max() < 1e-5

    def test_gradient_flows_backbone(self, gene_exp, edge_index, xyz, subgraph_id):
        backbone = GNNBackbone(
            gene_ch=N_GENES, spatial_ch=N_SPATIAL, hid_ch=HID_CH,
        )
        backbone.train()
        out = backbone(gene_exp, edge_index, xyz, subgraph_id)
        loss = out.sum()
        loss.backward()
        # Check that at least one parameter received a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in backbone.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# Task head tests
# ---------------------------------------------------------------------------
class TestTaskHeads:
    def test_classification_head_shape(self):
        head = ClassificationHead(in_ch=HID_CH, n_classes=N_CLASSES)
        x = torch.randn(N_NODES, HID_CH)
        out = head(x)
        assert out.shape == (N_NODES, N_CLASSES)

    def test_reconstruction_head_shape(self):
        head = ReconstructionHead(in_ch=HID_CH, n_genes=N_GENES)
        x = torch.randn(N_NODES, HID_CH)
        out = head(x)
        assert out.shape == (N_NODES, N_GENES)


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_full_pipeline(self, gene_exp, edge_index, xyz, subgraph_id):
        from gfs.models.feature_selection import GumbelFeatureSelector

        fs = GumbelFeatureSelector(n_genes=N_GENES, n_select=10)
        backbone = GNNBackbone(
            gene_ch=N_GENES, spatial_ch=N_SPATIAL, hid_ch=HID_CH,
        )
        head = ClassificationHead(in_ch=HID_CH, n_classes=N_CLASSES)

        fs.train()
        backbone.train()
        masked = fs(gene_exp, tau=1.0, subgraph_id=subgraph_id)
        embeddings = backbone(masked, edge_index, xyz, subgraph_id)
        logits = head(embeddings)

        assert logits.shape == (N_NODES, N_CLASSES)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.randint(0, N_CLASSES, (N_NODES,))
        )
        loss.backward()

    def test_gradient_flows_full_pipeline(self, gene_exp, edge_index, xyz, subgraph_id):
        from gfs.models.feature_selection import GumbelFeatureSelector

        fs = GumbelFeatureSelector(n_genes=N_GENES, n_select=10)
        backbone = GNNBackbone(
            gene_ch=N_GENES, spatial_ch=N_SPATIAL, hid_ch=HID_CH,
        )
        head = ClassificationHead(in_ch=HID_CH, n_classes=N_CLASSES)

        fs.train()
        backbone.train()
        head.train()

        masked = fs(gene_exp, tau=1.0, subgraph_id=subgraph_id)
        embeddings = backbone(masked, edge_index, xyz, subgraph_id)
        logits = head(embeddings)

        loss = torch.nn.functional.cross_entropy(
            logits, torch.randint(0, N_CLASSES, (N_NODES,))
        )
        loss.backward()

        # Gradients must flow all the way back to the feature selector logits
        assert fs.logits.grad is not None
        assert fs.logits.grad.abs().sum() > 0
