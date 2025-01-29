import warnings

import anndata as ad
import lightning as L
import numpy as np
import torch
from anndata._core.aligned_df import ImplicitModificationWarning
from scipy.sparse import issparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data as PyGData
from torch_geometric.loader.neighbor_loader import NeighborLoader

from gfs.utils import get_paths

warnings.filterwarnings("ignore", category=ImplicitModificationWarning, message="Transforming to str index.")


class PyGAnnDataGraphDataModule(L.LightningDataModule):
    """
    Data module using PyG functions to return graph patches.
    """

    def __init__(
        self,
        data_dir: None,
        file_names: list[str] = ["VISp_nhood.h5ad"],
        batch_size: int = 1,
        n_hops: int = 2,
        split_method: str = "rand",
        cell_type: str = "subclass",
        spatial_coords: list[str] = ["x_section", "y_section", "z_section"],
    ):
        super().__init__()
        if data_dir is None:
            data_dir = get_paths()["data_root"]
        self.adata_paths = [str(data_dir) + file_name for file_name in file_names]
        self.batch_size = batch_size
        self.n_hops = n_hops
        self.split_method = split_method
        self.cell_type = cell_type
        self.spatial_coords = spatial_coords

    def setup(self, stage: str):
        # including self.dataset for debugging.
        # consider removing this if we run into cpu memory limits.
        self.dataset = PyGAnnData(self.adata_paths, cell_type=self.cell_type, spatial_coords=self.spatial_coords)
        self.data = self.dataset.get_pygdata_obj()

    def train_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32,
            input_nodes=self.data.train_mask,
        )

    def val_dataloader(self):
        return NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def test_dataloader(self):
        return NeighborLoader(
            self.data,
            input_nodes=self.data.test_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def predict_dataloader(self):
        return NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )


def read_check_h5ad(path):
    adata = ad.read_h5ad(path)
    for field in ["spatial_connectivities", "spatial_distances"]:
        assert field in adata.obsp.keys(), f"{field} absent: Run `sc.pp.neighbors` first for {path}"
    return adata


def get_non_blank_genes(adata):
    keep_genes = adata.var[~adata.var.index.str.startswith("Blank")].index
    return keep_genes


class PyGAnnData:
    """
    Class to preprocess and build the PyG Data object from AnnData dataset

    Args:
        path (str): Path to the AnnData file.
        keep_genes (list): List of genes to keep.
        keep_cells (list): List of cells to keep.
        spatial_coords (list): Anndata.obs columns for spatial coordinates (for regression).
        cell_type (str): Anndata.obs column for cell types.
        max_order (int): Maximum order of neighbors to consider.
        d_threshold (float): Distance threshold (in mm) for considering neighbors.
    """

    def __init__(
        self,
        paths=[],
        keep_genes=None,
        keep_cells=None,
        spatial_coords=["x_ccf", "y_ccf", "z_ccf"],
        cell_type="supertype",
        max_order=2,
        d_threshold=1000,
        rand_seed=42,
    ):
        super().__init__()
        self.paths = paths

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
        self.max_order = max_order
        self.d_threshold = d_threshold

        # create binary adjacency matrix without self-loops
        adj = self.adata.obsp["spatial_connectivities"].copy()
        adj = adj.astype(bool).astype(int)
        # this is an extra precaution to prevent far connections.
        adj[self.adata.obsp["spatial_distances"] > self.d_threshold] = 0
        adj.setdiag(0)
        self.adj = adj

        self.spatial_coords = spatial_coords
        self.cell_type = cell_type
        self.cell_type_list = adata.obs[cell_type].cat.categories.tolist()
        self.cell_type_labelencoder = LabelEncoder()
        self.cell_type_labelencoder.fit(self.cell_type_list)
        self.data_issparse = issparse(adata.X)

        # reproducible train/val/test split for crossvalidation 5 folds using StratifiedKFold
        self.cv = 0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        splits = skf.split(self.adata, self.adata.obs[self.cell_type])
        splits = list(splits)
        self.train_ind = splits[self.cv][0]
        self.val_ind = splits[self.cv][1]

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

        # boolean mask for train/val/test
        # first, define all false
        train_mask = torch.zeros(self.adata.shape[0], dtype=torch.bool)
        train_mask[self.train_ind] = True
        val_mask = torch.zeros(self.adata.shape[0], dtype=torch.bool)
        val_mask[self.val_ind] = True

        return PyGData(
            gene_exp=gene_exp,
            edge_index=edgelist,
            xyz=xyz,
            celltype=celltype,
            num_nodes=gene_exp.shape[0],
            train_mask=train_mask,
            val_mask=val_mask,
        )


def test_pyganndatagraphdatamodule():
    import numpy as np

    from gfs.data.hemisphere import PyGAnnDataGraphDataModule
    from gfs.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=path, file_names=["test_one_section_hemi.h5ad"], batch_size=1, n_hops=2
    )
    datamodule.setup(stage="fit")
    dataloader = iter(datamodule.train_dataloader())

    for i in range(3):
        batch = next(dataloader)

        # 2-hop neighborhood from NeighborLoader:
        nhood = batch.n_id.numpy()
        nhood = list(set(nhood))
        nhood.sort()

        # calculate 2-hop neighborhood directly:
        ref_cell = batch.input_id.numpy()  # cell index from which 2-hop neighborhood is calculated.
        nhood_ = np.where(datamodule.dataset.adata.obsp["spatial_connectivities"][ref_cell, :].toarray())[1]
        nhood_ = set(nhood_)
        for i in nhood_:
            ref_cell_nhood_2 = np.where(datamodule.dataset.adata.obsp["spatial_connectivities"][i, :].toarray())[1]
            ref_cell_nhood_2 = set(ref_cell_nhood_2)
            nhood_ = nhood_.union(ref_cell_nhood_2)
        nhood_ = list(nhood_)
        nhood_.sort()

        assert len(set(nhood_) - set(nhood)) == 0, "Difference between ref_cell_nhood and nhood is not empty"

    print("anndatagraphdatamodule checks passed")

    return


def test_pyganndata():
    from gfs.data.hemisphere import PyGAnnData
    from gfs.utils import get_paths

    paths = get_paths()
    pygdata = PyGAnnData(
        paths=[paths["data_root"] + "test_one_section_hemi.h5ad", paths["data_root"] + "test_one_section_hemi.h5ad"],
        keep_genes=None,
        keep_cells=None,
        spatial_coords=["x_ccf", "y_ccf", "z_ccf"],
        cell_type="supertype",
        max_order=2,
        d_threshold=1000,
    )
    print("pyganndata checks passed")

    return pygdata


if __name__ == "__main__":
    print("running pyganndata")
    test_pyganndata()
    print("running pyganndata")
    test_pyganndatagraphdatamodule()
