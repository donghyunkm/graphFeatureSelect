"""Tabular datamodule for frozen-gene-panel baseline evaluation.

Serves the upsampled train set written by ``gfs.data.node_exposure`` plus the plain val/test
splits, as PyG ``Data`` batches carrying **self-edges only**. The batches are shaped so
``gfs.models.antelope.LitGnnFs`` consumes them unchanged -- same backbone, same training and
eval code, no graph.

Why self-edges rather than no ``edge_index`` at all: the GAT/SAGE/GCN conv layers still run, and
with a single self-edge and ``add_self_loops=False`` the attention softmax is over one element
(= 1.0), so the conv path reduces to a clean linear map of the cell's own expression. That is
the intended baseline architecture -- see the "Baseline eval protocol" notes in CLAUDE.md.

Parity with the graph run (measured, see ``gfs.data.node_exposure``):

| quantity                     | graph run          | this datamodule                |
|------------------------------|--------------------|--------------------------------|
| rows contributing loss/epoch | 4,839,573          | 4,839,401 (the upsampled set)  |
| rows per optimizer step      | ~4,147 (patch)     | ``batch_size`` (default 4147)  |
| optimizer steps/epoch        | 1,167              | 1,167                          |
| per-row weight in batch mean | 1/4,147            | 1/4,147                        |
| gradient weight/cell/epoch   | 0.015637           | 0.015637                       |

The graph run's per-cell repetition comes from patch overlap; here it is baked into the
upsampled dataset. ``CrossEntropyLoss`` is mean-reduced in both, so matching rows-per-step and
steps-per-epoch matches the gradient exactly.

val/test are **not** upsampled -- ``validation_step``/``test_step`` score root nodes only
(``antelope.py:472,565``), so each val/test cell already contributes exactly once per epoch.

The batch fields ``LitGnnFs`` requires, and what they are here:
    x, celltype, edge_index, num_nodes   as usual; edge_index is ``[[0..n-1],[0..n-1]]``
    gene_exp_ind / xyz_ind               index the panel and xyz blocks of ``x``
    subgraph_id                          ``arange(n)`` -- every row is its own patch, so
                                         per-patch xyz centring stays per-cell
    train_mask                           all-ones, matching the train graph
    n_id / input_id                      identity, so ``torch.where(n_id == input_id...)``
                                         selects every row in val/test/predict steps
"""

import anndata as ad
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data as PyGData


def _read_matrix(adata):
    x = adata.X
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


class PanelDataset(Dataset):
    """Rows of a gene panel plus label and xyz. Indexing returns a plain int; collation builds
    the PyG batch, so ``num_workers`` shards row indices rather than copying tensors."""

    def __init__(self, x, y, xyz):
        self.x = x
        self.y = y
        self.xyz = xyz

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return i


class PanelCollator:
    """Collate row indices into a self-edge PyG batch shaped like a NeighborLoader batch."""

    def __init__(self, dataset, n_genes):
        self.ds = dataset
        self.n_genes = n_genes

    def __call__(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        n = idx.numel()
        gene_exp = self.ds.x[idx]
        xyz = self.ds.xyz[idx]
        ar = torch.arange(n, dtype=torch.long)

        return PyGData(
            x=torch.cat([gene_exp, xyz], dim=1),
            # self-edges only: one edge per node, from and to itself
            edge_index=torch.stack([ar, ar], dim=0),
            celltype=self.ds.y[idx],
            xyz=xyz,
            num_nodes=n,
            gene_exp_ind=torch.arange(self.n_genes),
            xyz_ind=torch.arange(self.n_genes, self.n_genes + xyz.shape[1]),
            # every row is its own patch, so per-patch xyz centring stays per-cell
            subgraph_id=ar,
            train_mask=torch.ones(n, dtype=torch.bool),
            # identity n_id/input_id: `where(n_id == input_id.unsqueeze(-1))` selects all rows
            n_id=ar,
            input_id=ar,
        )


class PanelTabularDataModule(L.LightningDataModule):
    """Upsampled tabular train + plain val/test, as self-edge PyG batches.

    Args:
        exposure_dir: dir written by ``gfs.data.node_exposure`` (holds ``upsampled_train.h5ad``,
            ``tabular_val.h5ad``, ``tabular_test.h5ad``, ``classes.json``).
        path_valtest: source valtest h5ad, used to pull xyz for the val/test splits (the
            ``tabular_*.h5ad`` files carry labels but not spatial coords).
        batch_size: rows per optimizer step. Default 4147 = the graph run's mean patch size,
            which yields 1167 steps/epoch -- exact gradient parity.
        n_labels: number of classes; must match the train-derived encoding (22 here).
    """

    def __init__(
        self,
        exposure_dir,
        path_train,
        path_valtest,
        spatial_coords=("rec_ccf_x", "rec_ccf_y", "rec_ccf_z"),
        cell_type="subclass",
        batch_size=4147,
        num_workers=8,
        n_labels=22,
        rand_seed=42,
        drop_last=False,
    ):
        super().__init__()
        self.exposure_dir = exposure_dir
        self.path_train = path_train
        self.path_valtest = path_valtest
        self.spatial_coords = list(spatial_coords)
        self.cell_type = cell_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_labels = n_labels
        self.rand_seed = rand_seed
        self.drop_last = drop_last

    def _load_split(self, path, xyz_source=None, mask_name=None):
        adata = ad.read_h5ad(path)
        x = torch.from_numpy(_read_matrix(adata))
        y = torch.as_tensor(np.ascontiguousarray(adata.obs["label"].to_numpy()), dtype=torch.long)
        assert y.min() >= 0, f"{path} has unencoded labels; regenerate with gfs.data.node_exposure"
        assert y.max() < self.n_labels, f"{path} label {y.max().item()} >= n_labels {self.n_labels}"

        if xyz_source is None:
            # upsampled train: xyz follows the repeated rows via orig_index
            orig = adata.obs["orig_index"].to_numpy()
            xyz = torch.from_numpy(np.asarray(self.train_xyz_source[orig], dtype=np.float32))
        else:
            mask = xyz_source.obs[mask_name].to_numpy(dtype=bool)
            xyz = torch.from_numpy(np.asarray(xyz_source.obs.loc[mask, self.spatial_coords].values, dtype=np.float32))
            assert xyz.shape[0] == x.shape[0], f"{mask_name} xyz/x row mismatch"
        return PanelDataset(x, y, xyz)

    def setup(self, stage):
        d = self.exposure_dir

        # xyz for the upsampled train rows has to come from the source train h5ad, since
        # upsampled_train.h5ad carries only labels + provenance in obs
        import json
        from pathlib import Path

        meta = json.loads((Path(d) / "classes.json").read_text())
        self.classes = meta["classes"]
        assert len(self.classes) == self.n_labels, (
            f"classes.json has {len(self.classes)} classes but n_labels={self.n_labels}"
        )

        train_src = ad.read_h5ad(self.path_train)
        self.train_xyz_source = train_src.obs[self.spatial_coords].values
        self.n_genes = None

        self.ds_train = self._load_split(str(Path(d) / "upsampled_train.h5ad"))
        valtest = ad.read_h5ad(self.path_valtest)
        self.ds_val = self._load_split(str(Path(d) / "tabular_val.h5ad"), valtest, "val_mask")
        self.ds_test = self._load_split(str(Path(d) / "tabular_test.h5ad"), valtest, "test_mask")

        self.n_genes = self.ds_train.x.shape[1]
        for name, ds in [("val", self.ds_val), ("test", self.ds_test)]:
            assert ds.x.shape[1] == self.n_genes, f"{name} has {ds.x.shape[1]} genes, train has {self.n_genes}"

        print(
            f"panel-tabular: train {tuple(self.ds_train.x.shape)} "
            f"val {tuple(self.ds_val.x.shape)} test {tuple(self.ds_test.x.shape)} | "
            f"{self.n_genes} genes, {self.n_labels} classes"
        )
        steps = len(self.ds_train) // self.batch_size if self.drop_last else -(-len(self.ds_train) // self.batch_size)
        print(f"  batch_size {self.batch_size} -> {steps} optimizer steps/epoch (graph run: 1167 steps of ~4147 rows)")

    def _loader(self, ds, shuffle, drop_last=False):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=PanelCollator(ds, self.n_genes),
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
            generator=torch.Generator().manual_seed(self.rand_seed),
        )

    def train_dataloader(self):
        return self._loader(self.ds_train, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return self._loader(self.ds_val, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.ds_test, shuffle=False)
