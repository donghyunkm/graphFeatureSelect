"""Frozen gene panel with the graph KEPT — isolates noise-robustness from annealing.

The tabular panel arm (``antelope_panel.py``) removes the selector *and* the graph together, so a
low score there is ambiguous between "the soft-mask curriculum did the work" and "training against
noisy random neighbours regularized the representations". This trainer keeps the shuffled graph and
removes only the selector, separating the two.

Differs from ``antelope_shuffle.py`` in **exactly one way**: ``fs_method: "none"``, so no ``logits``
parameter exists, there is no soft-mask annealing and no per-patch Gumbel gene-dropout. Same loader,
same batch size (64), same 2-hop patches, same ``xyz_status: true``, same ``halfhop: true``, same
100 epochs. Per-cell exposure is the graph's own ~64.84 loss terms/epoch by construction, so no
upsampling is involved in this arm.

Reading the result against the shuffle GNN's 0.627 and the tabular baseline's ~0.57:
  ~0.62 -> the noisy conv path was doing the work (noise-robustness).
  ~0.57 -> the conv path contributes nothing; annealing/co-adaptation owns the gap.

Run (from scripts/):
    python -u ../gfs/trainers/antelope_panel_graph.py \\
        data.genes="[2,12,83,435,480]" data.prefix="panelgraph_shuffle" data.rand_seed=0
    # intact-graph control:
    python -u ../gfs/trainers/antelope_panel_graph.py \\
        data.shuffle_cells=false data.prefix="panelgraph_intact" data.rand_seed=0
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

from gfs.data.hemisphere_new_panel import PanelGraphDataModule
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
    """Print the panel's gene names, so the log records which panel actually ran."""
    src = ad.read_h5ad(paths["data_root"] + config.data.path_train, backed="r")
    genes = list(config.data.genes)
    assert len(genes) == config.data.n_genes, f"n_genes={config.data.n_genes} but genes has {len(genes)} entries"
    assert max(genes) < src.shape[1], f"gene index {max(genes)} out of range for {src.shape[1]} genes"
    print(f"panel: columns {genes} -> {src.var_names[genes].tolist()}")
    print(f"shuffle_cells={config.data.shuffle_cells}  (true => noisy neighbours, false => intact graph)")


@hydra.main(config_path="../configs", config_name="antelope_panel_graph", version_base="1.2")
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

    datamodule = PanelGraphDataModule(
        data_dir=None,
        path_train=paths["data_root"] + config.data.path_train,
        path_valtest=paths["data_root"] + config.data.path_valtest,
        batch_size=config.data.batch_size,
        n_hops=config.data.n_hops,
        cell_type=config.data.cell_type,
        spatial_coords=config.data.spatial_coords,
        self_loops_only=config.data.self_loops_only,
        rand_seed=config.data.rand_seed,
        genes=list(config.data.genes),
        shuffle_cells=config.data.shuffle_cells,
        num_workers=config.data.num_workers,
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
