"""Frozen-gene-panel baseline evaluation: no graph (self-edges), no selection module.

Trains the *same* ``LitGnnFs`` backbone as the graph arm on an already-selected gene panel, using
the upsampled tabular dataset from ``gfs.data.node_exposure`` so per-cell exposure and gradient
updates match the graph run. Importing ``LitGnnFs`` rather than copying it means the backbone
cannot drift from ``antelope.py``.

What differs from ``antelope.py``, and why:
  - ``fs_method: "none"`` -- no ``logits`` parameter exists, so nothing selection-related trains
    and no soft-mask warm start occurs. The panel is the input.
  - self-edges only -- with ``add_self_loops=False`` the attention softmax is over one element
    (= 1.0), so the conv path is a clean linear map of the cell's own expression.
  - ``batch_size: 4147`` on the upsampled set -- 1167 optimizer steps/epoch with 1/4147 per-row
    weight, identical to the graph run (see ``gfs/data/panel_tabular.py`` for the parity table).
  - no ``selections.csv`` is written (nothing is being selected); ``test_pred.pt`` still is.

Run (from scripts/):
    python -u ../gfs/trainers/antelope_panel.py \
        data.exposure_dir="exposure/persist_seed0_select5" \
        data.prefix="panel_persist_s0" data.rand_seed=0
"""

import random

import anndata as ad
import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from gfs.data.panel_tabular import PanelTabularDataModule
from gfs.models.antelope import LitGnnFs
from gfs.utils import get_datetime, get_paths


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_panel(config, paths):
    """Assert the built dataset really holds the panel named in the config.

    Guards the failure mode that would silently invalidate the comparison: pointing at an
    ``exposure_dir`` built for a different panel than ``data.genes`` records.
    """
    exposure_path = paths["data_root"] + config.data.exposure_dir + "/upsampled_train.h5ad"
    built = ad.read_h5ad(exposure_path, backed="r")
    source = ad.read_h5ad(paths["data_root"] + config.data.path_train, backed="r")
    want = source.var_names[list(config.data.genes)].tolist()
    got = built.var_names.tolist()
    assert got == want, (
        f"panel mismatch: {config.data.exposure_dir} holds genes {got}, but data.genes="
        f"{list(config.data.genes)} means {want}. Rebuild with "
        f"`python -m gfs.data.node_exposure --genes {','.join(map(str, config.data.genes))} ...`"
    )
    assert built.shape[1] == config.data.n_genes, (
        f"n_genes={config.data.n_genes} but the dataset has {built.shape[1]} columns"
    )
    print(f"panel verified: columns {list(config.data.genes)} -> {got}")


@hydra.main(config_path="../configs", config_name="antelope_panel", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    seed_everything(config.data.rand_seed, workers=False)
    setup_seeds(config.data.rand_seed)

    paths = get_paths()
    expname = get_datetime(expname=f"{config.data.prefix}_s{config.data.rand_seed}")
    log_path = paths["data_root"] + f"logs/{expname}"
    checkpoint_path = paths["data_root"] + f"checkpoints/{expname}"

    check_panel(config, paths)

    tb_logger = TensorBoardLogger(save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_f1_overall",
        filename="{epoch}-{val_f1_overall:.2f}",
        mode="max",
        save_top_k=1,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    datamodule = PanelTabularDataModule(
        exposure_dir=paths["data_root"] + config.data.exposure_dir,
        path_train=paths["data_root"] + config.data.path_train,
        path_valtest=paths["data_root"] + config.data.path_valtest,
        spatial_coords=config.data.spatial_coords,
        cell_type=config.data.cell_type,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        n_labels=config.data.n_labels,
        rand_seed=config.data.rand_seed,
        drop_last=config.data.drop_last,
    )

    model = LitGnnFs(config)
    assert not hasattr(model.model, "logits"), "selection module present; fs_method must be 'none'"

    trainer = L.Trainer(
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        max_epochs=config.trainer.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="gpu",
        devices=1,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, ckpt_path="best", datamodule=datamodule, weights_only=False)


if __name__ == "__main__":
    main()
