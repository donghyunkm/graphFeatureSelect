"""Training entry point for GFSNet."""

import hydra
import lightning as L
from omegaconf import DictConfig

from gfs.data.datamodule import HemisphereDataModule
from gfs.models.lit_module import LitGnnFs


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.data.seed)

    # Data
    dm = HemisphereDataModule(
        train_path=cfg.data.train_path,
        test_path=getattr(cfg.data, 'test_path', None),
        cell_type_col=cfg.data.cell_type_col,
        spatial_cols=list(cfg.data.spatial_cols),
        min_cells=cfg.data.min_cells,
        val_frac=cfg.data.val_frac,
        batch_size=cfg.data.batch_size,
        n_hops=cfg.data.n_hops,
        num_workers=cfg.data.num_workers,
        seed=cfg.data.seed,
    )
    dm.setup("fit")

    # Model (needs data dimensions)
    model = LitGnnFs(cfg)
    model.setup_model(n_genes=dm.n_genes, n_classes=dm.n_classes)

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=1,
    )

    # Train
    trainer.fit(model, dm)

    # Test (if test data available)
    if dm.test_data is not None:
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
