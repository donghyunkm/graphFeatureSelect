import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from gfs.data.hemisphere import PyGAnnDataGraphDataModule
from gfs.models.antelope import GnnFs
from gfs.utils import get_datetime, get_paths


@hydra.main(config_path="../configs", config_name="gnn")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    # paths
    paths = get_paths()
    expname = get_datetime(expname=config.expname)
    log_path = paths["data_root"] + f"logs/{expname}"
    checkpoint_path = paths["data_root"] + f"checkpoints/{expname}"

    # helpers
    tb_logger = TensorBoardLogger(save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_overall_acc",
        filename="{epoch}-{val_overall_acc:.2f}",
        mode="max",
        save_top_k=1,
        every_n_epochs=1,
    )

    # data
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=paths["data_root"],
        file_names=config.data.file_names,
        cell_type=config.data.cell_type,
        spatial_coords=config.data.spatial_coords,
        batch_size=config.data.batch_size,
        n_hops=config.data.n_hops,
    )

    # model
    model = GnnFs(config)

    # fit wrapper
    trainer = L.Trainer(
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        max_epochs=config.trainer.max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
