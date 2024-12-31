# datasets.py from nichecompass
import warnings

import anndata as ad
from anndata._core.aligned_df import ImplicitModificationWarning
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp
import torch
from graphFeatureSelect.utils import get_paths
# from .utils import sparse_mx_to_sparse_tensor


class AnnDataGraphDataset(Dataset):
    """
    Tabular AnnData dataset.

    Args:
        path (str): Path to the AnnData file.
        keep_genes (list): List of genes to keep.
        keep_cells (list): List of cells to keep.
        spatial_coords (list): Anndata.obs columns for spatial coordinates (for regression)
        cell_type (str): Anndata.obs column for cell types.
        max_order (int): Maximum order of neighbors to consider.
        d_threshold (float): Distance threshold (in mm) for considering neighbors.
    """

    def __init__(
        self,
        paths,
        keep_genes=None,
        keep_cells=None,
        spatial_coords=["x_ccf", "y_ccf", "z_ccf"],
        cell_type="supertype",
        max_order=2,
        d_threshold=1000,
    ):
        super().__init__()
        self.paths = paths
        adata = ad.read_h5ad(self.paths[0])
        if len(self.paths) > 1:
            for i in range(1, len(self.paths)):
                adata = ad.concat([adata, ad.read_h5ad(self.paths[i])], axis = 0, join='inner', merge = 'same')
        assert (
            "connectivities" in adata.obsp.keys()
        ), "Spatial connectivities not found. Run `sc.pp.neighbors` first."
        assert "distances" in adata.obsp.keys(), "Spatial distances not found. Run `sc.pp.neighbors` first."

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
        adj = self.adata.obsp["connectivities"].copy()
        adj = adj.astype(bool).astype(int)
        adj[self.adata.obsp["distances"] > self.d_threshold] = 0 # the distances/connectivities are already thresholded by scanpy.pp.neighbors
        adj.setdiag(0)
        self.adj = adj

        self.spatial_coords = spatial_coords
        self.cell_type = cell_type
        self.cell_type_list = adata.obs[cell_type].cat.categories.tolist()
        self.cell_type_labelencoder = LabelEncoder()
        self.cell_type_labelencoder.fit(self.cell_type_list)
        self.data_issparse = issparse(adata.X)

        if self.data_issparse: 
            self.x = torch.tensor(adata.X.toarray())
        else:
            self.x = torch.tensor(adata.X)

        self.edge_index = self.convert_torch_sparse_coo(adj)

        self.labels = self.cell_type_labelencoder.transform(self.adata.obs.iloc[[i for i in range(self.adata.shape[0])]][self.cell_type])

    def convert_torch_sparse_coo(self, adj):
        csr = adj
        coo_matrix = csr.tocoo()
        values = coo_matrix.data
        indices = np.vstack((coo_matrix.row, coo_matrix.col))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        shape = coo_matrix.shape
        sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        edge_index = sparse_tensor._indices()
        return edge_index

    def celltypes(self):
        idx = [i for i in range(self.adata.shape[0])]
        return self.cell_type_labelencoder.transform(self.adata.obs.iloc[idx][self.cell_type])

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        idx = [idx]
        gene_exp = self.adata.X[idx, :]
        if self.data_issparse:
            gene_exp = (gene_exp.toarray().astype(np.float32))
        xyz = self.adata.obs.iloc[idx][self.spatial_coords].values.astype(np.float32)
        celltype = self.cell_type_labelencoder.transform(self.adata.obs.iloc[idx][self.cell_type])
        return gene_exp, celltype


def get_non_blank_genes(adata):
    keep_genes = adata.var[~adata.var.index.str.startswith("Blank")].index
    return keep_genes
