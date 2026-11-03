"""Frozen gene panel with 4-hop complete-field training — the feature-ablation arm.

Identical to ``antelope_4hop.py`` (4-hop sampling, ``trainmode=2`` complete-field loss, root-only
eval) except ``fs_method="none"``: no selection module, the frozen panel is the input. Compared
against the 4-hop *with-selection* runs (`main_fulldata_4hop_seeds.sh`) at matched settings, this
isolates the contribution of learned feature selection from the panel + backbone. See
``docs/feature_ablation.md``.

Run (from scripts/):
    python -u ../gfs/trainers/antelope_4hop_panel.py --config-name antelope_4hop_panel \\
        data.genes="[2,250,435,480,481]" data.prefix="full_4hop_panel_seed0" data.rand_seed=0
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

from gfs.data.hemisphere_new_4hop_panel import Panel4hopDataModule
from gfs.models.antelope import LitGnnFs
from gfs.utils import get_datetime, get_paths


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def report_panel(config, paths):
    """Print the panel's gene names so the log records which panel actually ran."""
    src = ad.read_h5ad(paths["data_root"] + config.data.path_train, backed="r")
    genes = list(config.data.genes)
    assert len(genes) == config.data.n_genes, f"n_genes={config.data.n_genes} but genes has {len(genes)} entries"
    assert max(genes) < src.shape[1], f"gene index {max(genes)} out of range for {src.shape[1]} genes"
    print(f"panel: columns {genes} -> {src.var_names[genes].tolist()}")


@hydra.main(config_path="../configs", config_name="antelope_4hop_panel", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    seed_everything(config.data.rand_seed, workers=False)
    setup_seeds(config.data.rand_seed)

    paths = get_paths()
    expname = get_datetime(expname=f"{config.data.prefix}_s{config.data.rand_seed}")
    log_path = paths["data_root"] + f"logs/{expname}"
    checkpoint_path = paths["data_root"] + f"checkpoints/{expname}"

    report_panel(config, paths)

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

    datamodule = Panel4hopDataModule(
        data_dir=paths["data_root"],
        path_train=paths["data_root"] + config.data.path_train,
        path_valtest=paths["data_root"] + config.data.path_valtest,
        cell_type=config.data.cell_type,
        spatial_coords=config.data.spatial_coords,
        self_loops_only=config.data.self_loops_only,
        batch_size=config.data.batch_size,
        n_hops=config.data.n_hops,
        loss_hops=config.data.loss_hops,
        rand_seed=config.data.rand_seed,
        genes=list(config.data.genes),
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
