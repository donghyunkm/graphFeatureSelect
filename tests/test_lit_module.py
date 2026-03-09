"""Test that LitGnnFs constructs from config and training_step runs with a mock batch."""

import torch
from hydra import compose, initialize_config_dir
from pathlib import Path
from torch_geometric.data import Data as PyGData
from unittest.mock import MagicMock

from gfs.models.lit_module import LitGnnFs


def _make_mock_batch(n_nodes=64, n_input=16, gene_ch=500, spatial_ch=3, n_labels=158):
    x = torch.randn(n_nodes, gene_ch + spatial_ch)
    edge_index = torch.randint(0, n_nodes, (2, n_nodes * 4))
    batch = PyGData(
        x=x,
        edge_index=edge_index,
        celltype=torch.randint(0, n_labels, (n_nodes,)),
        gene_exp_ind=torch.arange(gene_ch),
        xyz_ind=torch.arange(gene_ch, gene_ch + spatial_ch),
        subgraph_id=torch.zeros(n_nodes, dtype=torch.long),
        n_id=torch.arange(n_nodes),
        input_id=torch.arange(n_input),
        num_nodes=n_nodes,
    )
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_input] = True
    batch.train_mask = train_mask
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask[n_input:2*n_input] = True
    batch.val_mask = val_mask
    return batch


def test_lit_module_persist():
    config_dir = str(Path(__file__).parent.parent / "src" / "gfs" / "conf")
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="config", overrides=["model=antelope", "model.halfhop=false"])
        model = LitGnnFs(cfg)

        trainer_mock = MagicMock()
        trainer_mock.max_epochs = 500
        trainer_mock.current_epoch = 0
        model._trainer = trainer_mock

        batch = _make_mock_batch()
        model.train()
        loss = model.training_step(batch, 0)
        assert loss is not None
        assert loss.requires_grad
        print("LitGnnFs persist training_step OK")


def test_lit_module_stg():
    config_dir = str(Path(__file__).parent.parent / "src" / "gfs" / "conf")
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="config", overrides=["model=antelope_stg", "model.halfhop=false"])
        model = LitGnnFs(cfg)

        trainer_mock = MagicMock()
        trainer_mock.max_epochs = 500
        trainer_mock.current_epoch = 0
        model._trainer = trainer_mock

        batch = _make_mock_batch()
        model.train()
        loss = model.training_step(batch, 0)
        assert loss is not None
        assert loss.requires_grad
        print("LitGnnFs stg training_step OK")


if __name__ == "__main__":
    test_lit_module_persist()
    test_lit_module_stg()
    print("All LitGnnFs tests passed")
