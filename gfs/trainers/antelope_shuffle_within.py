"""``antelope_shuffle.py`` with the valtest permutation restricted to within-val / within-test.

Identical to ``gfs/trainers/antelope_shuffle.py`` in every respect except the datamodule import:
``hemisphere_new_shuffle_within`` instead of ``hemisphere_new_shuffle``. The train permutation,
the model, the seeding, the logger, the checkpoint monitor and the loaders are all unchanged, so a
run of this against a completed ``*_shuffle_s*`` run differs in exactly one thing.

See ``gfs/data/hemisphere_new_shuffle_within.py`` for why: the parent permutes valtest cells across
the val/test boundary, which re-partitions which cells constitute each split. This keeps val and
test holding exactly the cells the h5ad assigns them, while still destroying the spatial
arrangement inside each.
"""

import random

import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from gfs.data.hemisphere_new_shuffle_within import PyGAnnDataGraphDataModule
from gfs.models.antelope import LitGnnFs
from gfs.utils import get_datetime, get_paths


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="../configs", config_name="antelope", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    seed_everything(config.data.rand_seed, workers=False)
    setup_seeds(config.data.rand_seed)

    paths = get_paths()
    expname_config = f"{config.data.prefix}_s{config.data.rand_seed}"
    expname = get_datetime(expname=expname_config)
    log_path = paths["data_root"] + f"logs/{expname}"
    checkpoint_path = paths["data_root"] + f"checkpoints/{expname}"

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

    datamodule = PyGAnnDataGraphDataModule(
        data_dir=paths["data_root"],
        path_train=paths["data_root"] + config.data.path_train,
        path_valtest=paths["data_root"] + config.data.path_valtest,
        cell_type=config.data.cell_type,
        spatial_coords=config.data.spatial_coords,
        self_loops_only=config.data.self_loops_only,
        batch_size=config.data.batch_size,
        n_hops=config.data.n_hops,
        rand_seed=config.data.rand_seed,
    )

    model = LitGnnFs(config)

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
