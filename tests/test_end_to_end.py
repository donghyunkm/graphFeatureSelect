"""End-to-end integration tests for the full training pipeline."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

DEV_DIR = Path("data/dev")
TRAIN_PATH = DEV_DIR / "section_1199651094_original.h5ad"
TEST_PATH = DEV_DIR / "section_1199651094_reflected.h5ad"

pytestmark = pytest.mark.skipif(not DEV_DIR.exists(), reason="Dev data not found")


@pytest.fixture(scope="module")
def dm():
    """Set up DataModule with dev data."""
    from gfs.data.datamodule import HemisphereDataModule

    dm = HemisphereDataModule(
        train_path=str(TRAIN_PATH),
        test_path=str(TEST_PATH),
        batch_size=32,
        n_hops=1,
        num_workers=0,
    )
    dm.setup("fit")
    return dm


@pytest.fixture(scope="module")
def config():
    """Minimal config for testing."""
    return OmegaConf.create({
        "backbone": {
            "gnn_type": "sage",
            "hid_ch": 16,
            "n_layers": 1,
            "dropout": 0.0,
            "heads": 1,
            "pre_linear": True,
            "residual": True,
            "layer_norm": True,
            "batch_norm": False,
            "jk": False,
            "xyz_proj": False,
            "x_residual": False,
        },
        "feature_selection": {
            "method": "gumbel",
            "tautype": "constant",
        },
        "task": {"name": "classification", "loss": "ce", "focal_loss": False},
        "data": {"spatial_cols": ["center_x", "center_y"]},
        "trainer": {
            "lr": 0.001,
            "lr_scheduler": "constant",
            "max_epochs": 2,
            "limit_train_batches": 2,
            "limit_val_batches": 2,
        },
        "logging": {"on_step": False, "on_epoch": True, "prog_bar": False, "logger": False},
        "n_select": 5,
        "trainmode": 0,
        "halfhop": False,
        "lam": 0.1,
        "expname": "test",
    })


def test_model_creation(config, dm):
    """Model can be created and initialized with data dimensions."""
    from gfs.models.lit_module import LitGnnFs

    model = LitGnnFs(config)
    model.setup_model(n_genes=dm.n_genes, n_classes=dm.n_classes)

    assert model.n_genes == dm.n_genes
    assert model.n_classes == dm.n_classes
    assert model.feature_selector is not None
    assert model.backbone is not None
    assert model.task_head is not None


def test_single_training_step(config, dm):
    """Model can execute a single training step."""
    import lightning as L

    from gfs.models.lit_module import LitGnnFs

    model = LitGnnFs(config)
    model.setup_model(n_genes=dm.n_genes, n_classes=dm.n_classes)

    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dm)


def test_training_reduces_loss(config, dm):
    """Loss decreases over a few training steps."""
    from gfs.models.lit_module import LitGnnFs

    model = LitGnnFs(config)
    model.setup_model(n_genes=dm.n_genes, n_classes=dm.n_classes)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        pred = model.forward(
            batch.gene_exp,
            batch.edge_index,
            batch.xyz,
            subgraph_id=None,
            tau=0.1,
        )
        loss = torch.nn.functional.cross_entropy(pred[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease (at least first vs last)
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_val_uses_hard_masks(config, dm):
    """At eval time, feature selector produces binary masks."""
    from gfs.models.lit_module import LitGnnFs

    model = LitGnnFs(config)
    model.setup_model(n_genes=dm.n_genes, n_classes=dm.n_classes)
    model.eval()

    mask = model.feature_selector.get_mask()
    assert ((mask == 0) | (mask == 1)).all(), "Eval mask is not binary"


def test_all_feature_selectors(dm):
    """All three feature selection methods work end-to-end."""
    import lightning as L
    from omegaconf import OmegaConf

    from gfs.models.lit_module import LitGnnFs

    for method in ["gumbel", "stg", "scgist"]:
        cfg = OmegaConf.create({
            "backbone": {
                "gnn_type": "sage",
                "hid_ch": 16,
                "n_layers": 1,
                "dropout": 0.0,
                "heads": 1,
                "pre_linear": True,
                "residual": True,
                "layer_norm": True,
                "batch_norm": False,
                "jk": False,
                "xyz_proj": False,
                "x_residual": False,
            },
            "feature_selection": {"method": method, "tautype": "constant", "sigma": 0.5, "l1": 0.1},
            "task": {"name": "classification", "loss": "ce", "focal_loss": False},
            "data": {"spatial_cols": ["center_x", "center_y"]},
            "trainer": {
                "lr": 0.001,
                "lr_scheduler": "constant",
                "max_epochs": 1,
                "limit_train_batches": 1,
                "limit_val_batches": 0,
            },
            "logging": {"on_step": False, "on_epoch": True, "prog_bar": False, "logger": False},
            "n_select": 5,
            "trainmode": 0,
            "halfhop": False,
            "lam": 0.1,
            "expname": "test",
        })

        model = LitGnnFs(cfg)
        model.setup_model(n_genes=dm.n_genes, n_classes=dm.n_classes)

        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, dm)


def test_reconstruction_head(dm):
    """Reconstruction task head works end-to-end."""
    import lightning as L
    from omegaconf import OmegaConf

    from gfs.models.lit_module import LitGnnFs

    cfg = OmegaConf.create({
        "backbone": {
            "gnn_type": "sage",
            "hid_ch": 16,
            "n_layers": 1,
            "dropout": 0.0,
            "heads": 1,
            "pre_linear": True,
            "residual": True,
            "layer_norm": True,
            "batch_norm": False,
            "jk": False,
            "xyz_proj": False,
            "x_residual": False,
        },
        "feature_selection": {"method": "gumbel", "tautype": "constant"},
        "task": {"name": "reconstruction", "hidden": [64]},
        "data": {"spatial_cols": ["center_x", "center_y"]},
        "trainer": {
            "lr": 0.001,
            "lr_scheduler": "constant",
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 0,
        },
        "logging": {"on_step": False, "on_epoch": True, "prog_bar": False, "logger": False},
        "n_select": 5,
        "trainmode": 0,
        "halfhop": False,
        "lam": 0.1,
        "expname": "test",
    })

    model = LitGnnFs(cfg)
    model.setup_model(n_genes=dm.n_genes, n_classes=dm.n_classes)

    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dm)
