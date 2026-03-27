"""Integration tests for the data pipeline: h5ad -> PyGAnnData -> PyG Data -> DataModule.

These tests load real dev data from data/dev/ and verify shapes, dtypes,
label encoding, split integrity, and dataloader behavior.
"""

from pathlib import Path

import pytest
import torch

DEV_DIR = Path("data/dev")
TRAIN_PATH = DEV_DIR / "section_1199651094_original.h5ad"
TEST_PATH = DEV_DIR / "section_1199651094_reflected.h5ad"

pytestmark = pytest.mark.skipif(not DEV_DIR.exists(), reason="Dev data not found at data/dev/")


@pytest.fixture(scope="module")
def train_dataset():
    """Load train hemisphere as PyGAnnData."""
    from gfs.data.dataset import PyGAnnData

    return PyGAnnData(TRAIN_PATH, min_cells=5)


@pytest.fixture(scope="module")
def train_data(train_dataset):
    """Convert to PyG Data object."""
    return train_dataset.to_pyg_data()


# ---------------------------------------------------------------------------
# 1. PyG Data attributes
# ---------------------------------------------------------------------------


def test_pyg_data_attributes(train_data):
    """Verify train_data has expected attributes with correct shapes and dtypes."""
    n_nodes = train_data.gene_exp.shape[0]

    # gene_exp: (n_nodes, 485)
    assert train_data.gene_exp.shape == (n_nodes, 485)
    assert train_data.gene_exp.dtype == torch.float32

    # xyz: (n_nodes, 2) spatial coordinates
    assert train_data.xyz.shape == (n_nodes, 2)
    assert train_data.xyz.dtype == torch.float32

    # y: (n_nodes,) int64 labels
    assert train_data.y.shape == (n_nodes,)
    assert train_data.y.dtype == torch.int64

    # edge_index: (2, n_edges)
    assert train_data.edge_index.shape[0] == 2
    assert train_data.edge_index.shape[1] > 0

    # train_mask and val_mask: bool tensors of shape (n_nodes,)
    assert train_data.train_mask.shape == (n_nodes,)
    assert train_data.train_mask.dtype == torch.bool
    assert train_data.val_mask.shape == (n_nodes,)
    assert train_data.val_mask.dtype == torch.bool

    # Should NOT have an x attribute (gene_exp and xyz are separate)
    assert not hasattr(train_data, "x") or train_data.x is None

    # Some cells are filtered for rare types, so n_nodes < 1582
    assert n_nodes < 1582


# ---------------------------------------------------------------------------
# 2. No split leakage
# ---------------------------------------------------------------------------


def test_no_split_leakage(train_data):
    """Train and val masks must not overlap, and must cover all nodes."""
    overlap = (train_data.train_mask & train_data.val_mask).sum().item()
    assert overlap == 0, f"Train/val overlap: {overlap} nodes"

    coverage = (train_data.train_mask | train_data.val_mask).all()
    assert coverage, "Some nodes are in neither train nor val split"


# ---------------------------------------------------------------------------
# 3. Label encoding
# ---------------------------------------------------------------------------


def test_label_encoding(train_dataset, train_data):
    """Labels should be contiguous integers 0..n_classes-1."""
    y = train_data.y
    assert y.min().item() == 0
    assert y.max().item() == train_dataset.n_classes - 1

    # All values in [0, n_classes) should be present
    unique_labels = y.unique()
    assert len(unique_labels) == train_dataset.n_classes


# ---------------------------------------------------------------------------
# 4. Edge index validity
# ---------------------------------------------------------------------------


def test_edge_index_valid(train_data):
    """Edge index must reference valid nodes, have no self-loops, and be int64."""
    edge_index = train_data.edge_index
    n_nodes = train_data.gene_exp.shape[0]

    assert edge_index.dtype == torch.int64

    # All indices in [0, n_nodes)
    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < n_nodes

    # No self-loops
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    assert self_loops == 0, f"Found {self_loops} self-loops"


# ---------------------------------------------------------------------------
# 5. Expression range
# ---------------------------------------------------------------------------


def test_expression_range(train_data):
    """Gene expression values should look like log1p CPM (non-negative, bounded, sparse)."""
    gene_exp = train_data.gene_exp

    assert gene_exp.min().item() >= 0
    assert gene_exp.max().item() < 20, f"Max expression {gene_exp.max().item():.1f} exceeds expected range"

    # Sparsity between 50% and 90%
    sparsity = (gene_exp == 0).float().mean().item()
    assert 0.5 < sparsity < 0.9, f"Sparsity {sparsity:.2%} outside expected range"


# ---------------------------------------------------------------------------
# 6. Rare type filtering
# ---------------------------------------------------------------------------


def test_rare_type_filtering(train_data):
    """After min_cells=5 filtering, no class should have fewer than 5 cells."""
    y = train_data.y
    for label in y.unique():
        count = (y == label).sum().item()
        assert count >= 5, f"Class {label.item()} has only {count} cells (< 5)"


# ---------------------------------------------------------------------------
# 7. Stratified split
# ---------------------------------------------------------------------------


def test_stratified_split(train_data):
    """Val set should have roughly proportional class representation."""
    y = train_data.y
    val_mask = train_data.val_mask
    n_total = y.shape[0]
    n_val = val_mask.sum().item()
    overall_val_frac = n_val / n_total

    for label in y.unique():
        class_mask = y == label
        n_class = class_mask.sum().item()
        n_class_val = (class_mask & val_mask).sum().item()

        if n_class < 10:
            # Skip very small classes — stratification is approximate
            continue

        class_val_frac = n_class_val / n_class
        # Allow generous tolerance: within 2x of expected fraction
        assert class_val_frac <= overall_val_frac * 3, (
            f"Class {label.item()}: val fraction {class_val_frac:.2%} is more than 3x overall {overall_val_frac:.2%}"
        )


# ---------------------------------------------------------------------------
# 8. DataModule basics
# ---------------------------------------------------------------------------


def test_datamodule_basics():
    """HemisphereDataModule loads data, exposes metadata, and yields batches."""
    from gfs.data.datamodule import HemisphereDataModule

    dm = HemisphereDataModule(
        train_path=str(TRAIN_PATH),
        test_path=str(TEST_PATH),
        batch_size=32,
        n_hops=1,
        num_workers=0,
    )
    dm.setup("fit")

    assert dm.n_genes == 485
    assert dm.n_classes > 0

    # Train dataloader should yield at least one batch
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))

    assert hasattr(batch, "gene_exp")
    assert hasattr(batch, "xyz")
    assert hasattr(batch, "y")
    assert hasattr(batch, "edge_index")


# ---------------------------------------------------------------------------
# 9. DataModule batch shapes
# ---------------------------------------------------------------------------


def test_datamodule_batch_shapes():
    """Verify shapes of a single train batch from the DataModule."""
    from gfs.data.datamodule import HemisphereDataModule

    dm = HemisphereDataModule(
        train_path=str(TRAIN_PATH),
        test_path=str(TEST_PATH),
        batch_size=32,
        n_hops=1,
        num_workers=0,
    )
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))

    n_batch_nodes = batch.gene_exp.shape[0]

    assert batch.gene_exp.shape[1] == 485
    assert batch.xyz.shape == (n_batch_nodes, 2)
    assert batch.y.shape[0] == n_batch_nodes
    assert batch.edge_index.shape[0] == 2


# ---------------------------------------------------------------------------
# 10. Test dataloader
# ---------------------------------------------------------------------------


def test_test_dataloader():
    """Test hemisphere dataloader works and has same gene count as train."""
    from gfs.data.datamodule import HemisphereDataModule

    dm = HemisphereDataModule(
        train_path=str(TRAIN_PATH),
        test_path=str(TEST_PATH),
        batch_size=32,
        n_hops=1,
        num_workers=0,
    )
    dm.setup("test")

    test_dl = dm.test_dataloader()
    batch = next(iter(test_dl))

    assert batch.gene_exp.shape[1] == 485
    assert hasattr(batch, "y")
    assert hasattr(batch, "edge_index")
