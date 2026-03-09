import random
import warnings

import anndata as ad
import lightning as L
import numpy as np
import torch
from anndata._core.aligned_df import ImplicitModificationWarning
from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data as PyGData
from torch_geometric.loader.neighbor_loader import NeighborLoader

from gfs.utils import get_paths

warnings.filterwarnings("ignore", category=ImplicitModificationWarning, message="Transforming to str index.")


class StratifiedKFold3(StratifiedKFold):

    def split(self, X, y, groups=None):
        s = super().split(X, y, groups)
        for train_indxs, test_indxs in s:
            y_train = y[train_indxs]
            train_indxs, cv_indxs = train_test_split(train_indxs,stratify=y_train, test_size=(1 / (self.n_splits - 1)))
            yield train_indxs, cv_indxs, test_indxs


def read_check_h5ad(path):
    adata = ad.read_h5ad(path)
    for field in ["spatial_connectivities"]:
        assert field in adata.obsp.keys(), f"{field} absent: Run `sklearn.neighbors.kneighbors_graph` first for {path}"
    return adata


def get_non_blank_genes(adata):
    keep_genes = adata.var[~adata.var.index.str.startswith("Blank")].index
    return keep_genes


class PyGAnnData:
    """
    Class to preprocess and build the PyG Data object from AnnData dataset

    Args:
        paths (list): List of paths to the AnnData files.
        keep_genes (list): List of genes to keep.
        keep_cells (list): List of cells to keep.
        spatial_coords (list): Anndata.obs columns for spatial coordinates (for regression).
        cell_type (str): Anndata.obs column for cell types.
        d_threshold (float): Distance threshold (in mm) for considering neighbors.
        rand_seed (int): Random seed for reproducibility of train/val split.
        test_data (bool): test dataset status
    """

    def __init__(
        self,
        paths=[],
        keep_genes=None,
        keep_cells=None,
        spatial_coords=["x_ccf", "y_ccf", "z_ccf"],
        cell_type="supertype",
        self_loops_only: bool = False,
        d_threshold=1000,
        n_splits=5,
        cv=0,
        rand_seed=42,
        test_data=False
    ):
        super().__init__()
        self.paths = paths
        self.test_data = test_data
        adata = read_check_h5ad(self.paths[0])
        if len(self.paths) > 1:
            for i in range(1, len(self.paths)):
                adata = ad.concat(
                    [adata, read_check_h5ad(self.paths[i])], axis=0, join="inner", merge="same", pairwise=True
                )

        # filter genes
        if keep_genes is not None:
            adata = adata[:, keep_genes].copy()
        else:
            keep_genes = get_non_blank_genes(adata)
            adata = adata[:, keep_genes].copy()

        # filter cells
        if keep_cells is not None:
            adata = adata[keep_cells, :].copy()

        self.adata = adata
        self.d_threshold = d_threshold

        # calculate binary adjacency matrix
        if self_loops_only:
            adj = csr_matrix(np.zeros((self.adata.shape[0], self.adata.shape[0]), dtype=int))
            adj.setdiag(1)
            self.adj = adj
        else:
            adj = self.adata.obsp["spatial_connectivities"].copy()
            adj = adj.astype(bool).astype(int)
            adj.setdiag(0)
            self.adj = adj

        self.cell_type = cell_type
        cell_type_col = adata.obs[self.cell_type]
        cell_type_counts = cell_type_col.value_counts()
        valid_cell_types = cell_type_counts[cell_type_counts >= 5].index
        mask = cell_type_col.isin(valid_cell_types)
        self.adata = adata[mask]

        self.spatial_coords = spatial_coords

        self.cell_type_list = adata.obs[cell_type].cat.categories.tolist()
        self.cell_type_labelencoder = LabelEncoder()
        self.cell_type_labelencoder.fit(self.cell_type_list)
        self.data_issparse = issparse(adata.X)

        # reproducible train/val/test split using StratifiedKFold
        self.cv = cv
        self.n_splits = n_splits
        assert self.cv < self.n_splits, "Crossvalidation index out of range"

        skf = StratifiedKFold3(n_splits=self.n_splits, shuffle=True, random_state=42)
        splits = skf.split(self.adata, self.adata.obs[self.cell_type])
        self.train_ind, self.val_ind, self.test_ind = list(splits)[self.cv]

    def convert_torch_sparse_coo(self, adj):
        coo_matrix = adj.tocoo()
        indices = np.vstack((coo_matrix.row, coo_matrix.col))
        values = coo_matrix.data
        shape = coo_matrix.shape

        indices = torch.tensor(indices, dtype=torch.int64)
        values = torch.tensor(values, dtype=torch.float32)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        return sparse_tensor._indices()

    def get_pygdata_obj(self):
        """Returns a `torch_geometric.data.Data` object.
        This implementation creates tensors from the full adata object.
        `edge_index` and `num_nodes` are required for NeighborLoader to work correctly."""

        if self.data_issparse:
            gene_exp = torch.tensor(self.adata.X.toarray()).float()
        else:
            gene_exp = torch.tensor(self.adata.X).float()

        edgelist = self.convert_torch_sparse_coo(self.adj)

        celltype = self.cell_type_labelencoder.transform(
            self.adata.obs.iloc[[i for i in range(self.adata.shape[0])]][self.cell_type]
        )
        celltype = torch.tensor(celltype).long()

        xyz = torch.tensor(self.adata.obs[self.spatial_coords].values).float()

        # boolean masks for train/val/test
        train_mask = torch.zeros(self.adata.shape[0], dtype=torch.bool)
        train_mask[self.train_ind] = True
        val_mask = torch.zeros(self.adata.shape[0], dtype=torch.bool)
        val_mask[self.val_ind] = True
        test_mask = torch.zeros(self.adata.shape[0], dtype=torch.bool)
        test_mask[self.test_ind] = True

        x = torch.cat([gene_exp, xyz], dim=1)
        gene_exp_ind = torch.arange(gene_exp.shape[1])
        xyz_ind = torch.arange(gene_exp.shape[1], gene_exp.shape[1] + xyz.shape[1])

        if self.test_data:
            val_mask = torch.ones(self.adata.shape[0], dtype=torch.bool)
            train_mask = torch.ones(self.adata.shape[0], dtype=torch.bool)

        return PyGData(
            x=x,
            edge_index=edgelist,
            celltype=celltype,
            xyz=xyz,
            num_nodes=gene_exp.shape[0],
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            gene_exp_ind=gene_exp_ind,
            xyz_ind=xyz_ind,
        )


class NeighborLoaderMod:
    """Custom NeighborLoader wrapper that modifies each batch before yielding."""

    def __init__(self, neighborloader_obj, n_hops):
        self.neighborloader_obj = neighborloader_obj
        self.n_hops = n_hops

    def __iter__(self):
        """Returns an iterator that includes subgraph_id field."""
        for batch in self.neighborloader_obj:
            batch.subgraph_id = torch.zeros(batch.x.size(0), dtype=torch.long)

            input_nodes = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
            for i, this_node in enumerate(input_nodes):
                nhood_list = [this_node.unsqueeze(-1)]
                for _ in range(self.n_hops):
                    these_edges = torch.any(torch.isin(batch.edge_index, nhood_list[-1]), dim=0)
                    nhood_list.append(batch.edge_index[:, these_edges].unique())
                node_inds = torch.cat(nhood_list, dim=0).unique()
                batch.subgraph_id[node_inds] = i

            yield batch

    def __len__(self):
        return len(self.neighborloader_obj)


class PyGAnnDataGraphDataModule(L.LightningDataModule):
    """
    Data module using PyG functions to return graph patches.
    """

    def __init__(
        self,
        data_dir: None,
        file_names: list[str] = ["VISp_nhood.h5ad"],
        test_names: list[str] = ["visp_54.h5ad"],
        batch_size: int = 1,
        n_hops: int = 2,
        cell_type: str = "subclass",
        spatial_coords: list[str] = ["x_section", "y_section", "z_section"],
        self_loops_only: bool = False,
        d_threshold: float = 1000,
        n_splits: int = 5,
        cv: int = 0,
        rand_seed: int = 42,
    ):
        super().__init__()
        if data_dir is None:
            data_dir = get_paths()["data_root"]
        self.adata_paths = [str(data_dir) + file_name for file_name in file_names]
        self.test_paths = [str(data_dir) + test_name for test_name in test_names]
        self.batch_size = batch_size
        self.n_hops = n_hops
        self.cell_type = cell_type
        self.spatial_coords = spatial_coords
        self.d_threshold = d_threshold
        self.n_splits = n_splits
        self.cv = cv
        self.rand_seed = rand_seed
        self.self_loops_only = self_loops_only

    def setup(self, stage: str):
        self.dataset = PyGAnnData(
            self.adata_paths,
            spatial_coords=self.spatial_coords,
            cell_type=self.cell_type,
            self_loops_only=self.self_loops_only,
            d_threshold=self.d_threshold,
            n_splits=self.n_splits,
            cv=self.cv,
            rand_seed=self.rand_seed,
            test_data=False
        )
        self.data = self.dataset.get_pygdata_obj()

    def train_dataloader(self):
        og = NeighborLoader(
            self.data,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            input_nodes=self.data.train_mask,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42)
        )
        return NeighborLoaderMod(og, self.n_hops)

    def val_dataloader(self):
        og = NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42)
        )
        return NeighborLoaderMod(og, self.n_hops)

    def test_dataloader(self):
        og = NeighborLoader(
            self.data,
            input_nodes=self.data.test_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42)
        )
        return NeighborLoaderMod(og, self.n_hops)

    def predict_dataloader(self):
        og = NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42)
        )
        return NeighborLoaderMod(og, self.n_hops)


def seed_worker(worker_id):
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)
