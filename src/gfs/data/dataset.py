"""PyG dataset from AnnData h5ad files with spatial graphs."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


class PyGAnnData:
    """Convert a single h5ad file with spatial graph into a PyG Data object.

    The h5ad file must contain:
        - `.X`: gene expression matrix (dense or sparse)
        - `.obs[cell_type_col]`: cell type labels
        - `.obs[spatial_cols]`: spatial coordinates
        - `.obsp['spatial_connectivities']`: precomputed spatial adjacency
    """

    def __init__(
        self,
        path: str | Path,
        cell_type_col: str = "AIT33_subclass",
        spatial_cols: list[str] | None = None,
        min_cells: int = 5,
        val_frac: float = 0.15,
        seed: int = 42,
        label_encoder: LabelEncoder | None = None,
    ):
        if spatial_cols is None:
            spatial_cols = ["center_x", "center_y"]

        self.path = Path(path)
        self.cell_type_col = cell_type_col
        self.spatial_cols = spatial_cols
        self.min_cells = min_cells
        self.val_frac = val_frac
        self.seed = seed

        # Load and process
        adata = ad.read_h5ad(self.path)

        # Dense float32 expression matrix
        X = adata.X
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Filter rare cell types
        cell_types = adata.obs[cell_type_col]
        counts = cell_types.value_counts()
        valid_types = counts[counts >= min_cells].index
        keep = cell_types.isin(valid_types).values

        if label_encoder is not None:
            # Reuse external encoder (test hemisphere): also drop unknown types
            known = cell_types.isin(label_encoder.classes_).values
            keep = keep & known

        adata = adata[keep].copy()
        X = X[keep]

        # Encode labels
        if label_encoder is not None:
            self.label_encoder = label_encoder
            labels = self.label_encoder.transform(adata.obs[cell_type_col])
        else:
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(adata.obs[cell_type_col])

        # Train/val/test masks
        n = len(labels)
        if label_encoder is not None:
            # External encoder → this is test data, all nodes are test
            train_mask = np.zeros(n, dtype=bool)
            val_mask = np.zeros(n, dtype=bool)
            test_mask = np.ones(n, dtype=bool)
        elif val_frac > 0:
            indices = np.arange(n)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_frac,
                stratify=labels,
                random_state=seed,
            )
            train_mask = np.zeros(n, dtype=bool)
            train_mask[train_idx] = True
            val_mask = np.zeros(n, dtype=bool)
            val_mask[val_idx] = True
            test_mask = np.zeros(n, dtype=bool)
        else:
            train_mask = np.ones(n, dtype=bool)
            val_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)

        # Store everything needed for to_pyg_data
        self._adata = adata
        self._X = X
        self._labels = labels
        self._train_mask = train_mask
        self._val_mask = val_mask
        self._test_mask = test_mask

        # Public metadata
        self.gene_names = list(adata.var_names)
        self.class_names = list(self.label_encoder.classes_)
        self.n_genes = len(self.gene_names)
        self.n_classes = len(self.class_names)

    def to_pyg_data(self) -> Data:
        """Build a PyG Data object from the loaded AnnData."""
        adata = self._adata

        gene_exp = torch.tensor(self._X, dtype=torch.float32)
        xyz = torch.tensor(
            adata.obs[self.spatial_cols].values.astype(np.float32),
            dtype=torch.float32,
        )
        y = torch.tensor(self._labels, dtype=torch.long)

        # Edge index from spatial adjacency
        adj = adata.obsp["spatial_connectivities"].copy()
        adj = (adj > 0).astype(float)
        adj.setdiag(0)
        adj.eliminate_zeros()
        edge_index = torch.tensor(np.stack(adj.nonzero()), dtype=torch.long)

        train_mask = torch.tensor(self._train_mask, dtype=torch.bool)
        val_mask = torch.tensor(self._val_mask, dtype=torch.bool)
        test_mask = torch.tensor(self._test_mask, dtype=torch.bool)

        n_nodes = gene_exp.shape[0]

        return Data(
            gene_exp=gene_exp,
            xyz=xyz,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=n_nodes,
            n_genes=self.n_genes,
            n_classes=self.n_classes,
        )
