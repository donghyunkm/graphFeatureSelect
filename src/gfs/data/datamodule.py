"""Lightning DataModule for inductive hemisphere split.

Train on one hemisphere h5ad, validate via random split within it,
and test on a separate hemisphere h5ad.
"""

from __future__ import annotations

import lightning as L
from torch_geometric.loader import NeighborLoader

from .dataset import PyGAnnData


class HemisphereDataModule(L.LightningDataModule):
    """DataModule for inductive hemisphere split.

    Loads a train hemisphere h5ad (with internal train/val split) and
    optionally a test hemisphere h5ad (all nodes used for testing).
    Dataloaders use PyG NeighborLoader for k-hop subgraph sampling.
    """

    def __init__(
        self,
        train_path: str,
        test_path: str | None = None,
        cell_type_col: str = "AIT33_subclass",
        spatial_cols: list[str] | None = None,
        min_cells: int = 5,
        val_frac: float = 0.15,
        batch_size: int = 64,
        n_hops: int = 2,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_path = train_path
        self.test_path = test_path
        self.cell_type_col = cell_type_col
        self.spatial_cols = spatial_cols or ["center_x", "center_y"]
        self.min_cells = min_cells
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.n_hops = n_hops
        self.num_workers = num_workers
        self.seed = seed

        # Populated by setup()
        self.train_ds: PyGAnnData | None = None
        self.test_ds: PyGAnnData | None = None
        self.train_data = None
        self.test_data = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if self.train_data is not None:
            return  # already set up

        # Train hemisphere: internal train/val split
        self.train_ds = PyGAnnData(
            path=self.train_path,
            cell_type_col=self.cell_type_col,
            spatial_cols=self.spatial_cols,
            min_cells=self.min_cells,
            val_frac=self.val_frac,
            seed=self.seed,
        )
        self.train_data = self.train_ds.to_pyg_data()

        # Test hemisphere (optional): reuse train label encoder, no split
        if self.test_path is not None:
            self.test_ds = PyGAnnData(
                path=self.test_path,
                cell_type_col=self.cell_type_col,
                spatial_cols=self.spatial_cols,
                min_cells=0,
                val_frac=0.0,
                seed=self.seed,
                label_encoder=self.train_ds.label_encoder,
            )
            self.test_data = self.test_ds.to_pyg_data()

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def _make_loader(self, data, mask, *, shuffle: bool) -> NeighborLoader:
        return NeighborLoader(
            data,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            input_nodes=mask,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> NeighborLoader:
        return self._make_loader(self.train_data, self.train_data.train_mask, shuffle=True)

    def val_dataloader(self) -> NeighborLoader:
        return self._make_loader(self.train_data, self.train_data.val_mask, shuffle=False)

    def test_dataloader(self) -> NeighborLoader:
        if self.test_data is None:
            raise RuntimeError("No test_path was provided — cannot create test dataloader.")
        return self._make_loader(self.test_data, self.test_data.test_mask, shuffle=False)

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def n_genes(self) -> int:
        """Number of gene features."""
        return self.train_ds.n_genes

    @property
    def n_classes(self) -> int:
        """Number of cell-type classes (from training data)."""
        return self.train_ds.n_classes

    @property
    def gene_names(self) -> list[str]:
        """Ordered list of gene names."""
        return self.train_ds.gene_names
