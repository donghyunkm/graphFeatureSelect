"""Per-epoch variant of the shuffled-cell control (``hemisphere_new_shuffle.py``): redraws the
expression/label permutation every epoch from a deterministic ``(run seed, epoch)`` seed, so the
conv path cannot memorize one fixed wrong neighborhood. xyz and edges are left intact. Empirically
equivalent to the fixed-permutation shuffle."""

import random
import warnings

import anndata as ad
import lightning as L
import numpy as np
import torch
from anndata._core.aligned_df import ImplicitModificationWarning
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data as PyGData
from torch_geometric.loader.neighbor_loader import NeighborLoader

from gfs.utils import get_paths

warnings.filterwarnings("ignore", category=ImplicitModificationWarning, message="Transforming to str index.")

# Multiplier used to decorrelate the per-epoch permutation seeds of neighbouring runs,
# so that run seed 0 / epoch 1 and run seed 1 / epoch 0 do not collide.
_EPOCH_SEED_STRIDE = 1_000_003


def epoch_shuffle_seed(rand_seed: int, epoch: int) -> int:
    """Deterministic permutation seed for a given (run seed, epoch) pair.

    Being a pure function of the epoch (rather than of a mutable counter) makes the
    shuffle idempotent: Lightning may call `train_dataloader()` more than once for the
    same epoch, and every such call must yield the same permutation.
    """
    return rand_seed * _EPOCH_SEED_STRIDE + epoch


def read_check_h5ad(path):
    adata = ad.read_h5ad(path)
    for field in ["spatial_connectivities"]:
        assert field in adata.obsp.keys(), f"{field} absent: Run `sklearn.neighbors.kneighbors_graph` first for {path}"
    return adata


class PyGAnnData:
    """
    Class to preprocess and build the PyG Data object from AnnData dataset.

    Per-epoch shuffling variant: gene expression and cell-type labels are permuted
    together, spatial coordinates and graph connectivity are left untouched, and the
    training permutation is redrawn at every epoch.

    Args:
        paths (list): List of paths to the AnnData files.
        spatial_coords (list): Anndata.obs columns for spatial coordinates (for regression).
        cell_type (str): Anndata.obs column for cell types.
        rand_seed (int): Random seed for reproducibility of train/val split.
    """

    def __init__(
        self,
        path_train,
        path_valtest,
        spatial_coords=["rec_ccf_x", "rec_ccf_y", "rec_ccf_z"],
        cell_type="supertype",
        self_loops_only: bool = False,
        rand_seed=42,
    ):
        super().__init__()
        adata_train = read_check_h5ad(path_train)
        adata_valtest = read_check_h5ad(path_valtest)

        self.adata_train = adata_train
        self.adata_valtest = adata_valtest
        self.rand_seed = rand_seed
        # calculate binary adjacency matrix
        if self_loops_only:
            # make a 'scipy.sparse._csr.csr_matrix' all zeros
            adj = csr_matrix(np.zeros((self.adata_train.shape[0], self.adata_train.shape[0]), dtype=int))
            adj.setdiag(1)
            self.adj_train = adj

            adj = csr_matrix(np.zeros((self.adata_valtest.shape[0], self.adata_valtest.shape[0]), dtype=int))
            adj.setdiag(1)
            self.adj_valtest = adj
        else:
            adj = self.adata_train.obsp["spatial_connectivities"].copy()
            adj = adj.astype(bool).astype(int)
            adj.setdiag(0)  # removing self-loops
            self.adj_train = adj

            adj = self.adata_valtest.obsp["spatial_connectivities"].copy()
            adj = adj.astype(bool).astype(int)
            adj.setdiag(0)  # removing self-loops
            self.adj_valtest = adj

        self.cell_type = cell_type
        self.spatial_coords = spatial_coords

        self.cell_type_list = adata_train.obs[cell_type].cat.categories.tolist()
        self.cell_type_labelencoder = LabelEncoder()
        self.cell_type_labelencoder.fit(self.cell_type_list)
        self.data_issparse = issparse(adata_train.X)

        # unshuffled reference copies, kept so that every epoch permutes the original
        # ordering rather than compounding permutations of the previous epoch.
        self.base_gene_exp_train = None
        self.base_celltype_train = None

    def convert_torch_sparse_coo(self, adj):
        coo_matrix = adj.tocoo()
        indices = np.vstack((coo_matrix.row, coo_matrix.col))
        values = coo_matrix.data
        shape = coo_matrix.shape

        indices = torch.tensor(indices, dtype=torch.int64)
        values = torch.tensor(values, dtype=torch.float32)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        return sparse_tensor._indices()

    def get_pygdata_obj_train(self, shuffle_seed=None):
        """Returns a `torch_geometric.data.Data` object.

        Gene-expression profiles and cell-type labels are shuffled together.
        Spatial coordinates and graph connectivity remain unchanged.

        This randomizes which biological cell is assigned to each fixed spatial
        graph node while preserving the alignment between gene expression and
        cell-type labels. `shuffle_seed` selects the permutation; pass a different
        value each epoch to redraw it (see `epoch_shuffle_seed`).
        """

        if self.data_issparse:
            gene_exp = torch.tensor(self.adata_train.X.toarray()).float()
        else:
            gene_exp = torch.tensor(self.adata_train.X).float()

        celltype = self.cell_type_labelencoder.transform(self.adata_train.obs[self.cell_type])
        celltype = torch.tensor(celltype).long()

        # keep the unpermuted ordering for the per-epoch reshuffles
        self.base_gene_exp_train = gene_exp.clone()
        self.base_celltype_train = celltype.clone()

        # Leave spatial coordinates unchanged
        xyz = torch.tensor(self.adata_train.obs[self.spatial_coords].values).float()

        # Leave graph connectivity unchanged
        edgelist = self.convert_torch_sparse_coo(self.adj_train).long()

        num_cells = gene_exp.shape[0]

        if shuffle_seed is None:
            shuffle_seed = self.rand_seed

        # Reproducible random permutation
        print(f"train shuffle (seed {shuffle_seed})")
        generator = torch.Generator()
        generator.manual_seed(shuffle_seed)
        perm = torch.randperm(num_cells, generator=generator)

        # Shuffle gene expression and labels using the same permutation
        gene_exp = gene_exp[perm]
        celltype = celltype[perm]

        # xyz and edgelist are intentionally not shuffled
        x = torch.cat([gene_exp, xyz], dim=1)

        gene_exp_ind = torch.arange(gene_exp.shape[1])
        xyz_ind = torch.arange(gene_exp.shape[1], gene_exp.shape[1] + xyz.shape[1])

        return PyGData(
            x=x,
            edge_index=edgelist,
            celltype=celltype,
            xyz=xyz,
            num_nodes=num_cells,
            gene_exp_ind=gene_exp_ind,
            xyz_ind=xyz_ind,
            train_mask=torch.ones(num_cells, dtype=torch.bool),
        )

    def reshuffle_train(self, data_train, shuffle_seed):
        """Redraws the training permutation in place on an existing `PyGData` object.

        Only the gene-expression block of `x` and the `celltype` labels are rewritten,
        both with the same permutation. `xyz` (and the `xyz` columns of `x`),
        `edge_index` and `train_mask` are left exactly as they were, so each biological
        cell lands on a different spatial node while the graph itself is untouched.

        Mutating in place avoids reallocating the ~n_cells x n_genes expression matrix
        once per epoch.
        """
        num_cells = self.base_gene_exp_train.shape[0]
        n_genes = self.base_gene_exp_train.shape[1]

        generator = torch.Generator()
        generator.manual_seed(shuffle_seed)
        perm = torch.randperm(num_cells, generator=generator)

        data_train.x[:, :n_genes] = self.base_gene_exp_train[perm]
        data_train.celltype.copy_(self.base_celltype_train[perm])

        print(f"train reshuffle (seed {shuffle_seed}), perm[:5] = {perm[:5].tolist()}")
        return data_train

    def get_pygdata_obj_valtest(self):
        """Returns a `torch_geometric.data.Data` object.

        Gene-expression profiles and cell-type labels are shuffled together.
        Spatial coordinates, graph connectivity, and validation/test node masks
        remain unchanged.

        This randomizes which biological cell is assigned to each fixed spatial
        graph node while preserving the alignment between gene expression and
        cell-type labels.

        Note: val/test use a single fixed permutation (from `rand_seed`) and are NOT
        reshuffled per epoch, so that `val_f1_overall` stays comparable across epochs
        for checkpoint selection and the reported test score is a stable measurement.
        """

        if self.data_issparse:
            gene_exp = torch.tensor(self.adata_valtest.X.toarray()).float()
        else:
            gene_exp = torch.tensor(self.adata_valtest.X).float()

        celltype = self.cell_type_labelencoder.transform(self.adata_valtest.obs[self.cell_type])
        celltype = torch.tensor(celltype).long()

        # Leave spatial coordinates unchanged
        xyz = torch.tensor(self.adata_valtest.obs[self.spatial_coords].values).float()

        # Leave graph connectivity unchanged
        edgelist = self.convert_torch_sparse_coo(self.adj_valtest).long()

        num_cells = gene_exp.shape[0]

        # Reproducible random permutation
        generator = torch.Generator()
        generator.manual_seed(self.rand_seed)
        perm = torch.randperm(num_cells, generator=generator)

        # Shuffle gene expression and cell-type labels together
        gene_exp = gene_exp[perm]
        celltype = celltype[perm]

        # xyz and edgelist are intentionally left unchanged
        x = torch.cat([gene_exp, xyz], dim=1)

        gene_exp_ind = torch.arange(gene_exp.shape[1])
        xyz_ind = torch.arange(gene_exp.shape[1], gene_exp.shape[1] + xyz.shape[1])

        # Leave validation and test node assignments unchanged
        val_mask = torch.tensor(self.adata_valtest.obs["val_mask"].to_numpy(dtype=bool), dtype=torch.bool)
        test_mask = torch.tensor(self.adata_valtest.obs["test_mask"].to_numpy(dtype=bool), dtype=torch.bool)

        return PyGData(
            x=x,
            edge_index=edgelist,
            celltype=celltype,
            xyz=xyz,
            num_nodes=num_cells,
            gene_exp_ind=gene_exp_ind,
            xyz_ind=xyz_ind,
            val_mask=val_mask,
            test_mask=test_mask,
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

            # start from each input_node and traverse outwards
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

    Per-epoch shuffling variant: the training permutation is redrawn every epoch.
    Requires `reload_dataloaders_every_n_epochs=1` on the Lightning `Trainer`, which is
    what causes `train_dataloader()` to be re-invoked at the start of each epoch.
    """

    def __init__(
        self,
        data_dir: None,
        path_train="/data/users1/dkim195/graphFeatureSelect/data/isocortex_train.h5ad",
        path_valtest="/data/users1/dkim195/graphFeatureSelect/data/isocortex_valtest.h5ad",
        batch_size: int = 1,
        n_hops: int = 2,
        cell_type: str = "subclass",
        spatial_coords: list[str] = ["rec_ccf_x", "rec_ccf_y", "rec_ccf_z"],
        self_loops_only: bool = False,
        rand_seed: int = 42,
    ):
        super().__init__()
        if data_dir is None:
            data_dir = get_paths()["data_root"]
        self.path_train = path_train
        self.path_valtest = path_valtest
        self.batch_size = batch_size
        self.n_hops = n_hops
        self.cell_type = cell_type
        self.spatial_coords = spatial_coords
        self.rand_seed = rand_seed
        self.self_loops_only = self_loops_only

    def setup(self, stage: str):
        # including self.dataset for debugging.
        # consider removing this if we run into cpu memory limits.
        self.dataset = PyGAnnData(
            path_train=self.path_train,
            path_valtest=self.path_valtest,
            spatial_coords=self.spatial_coords,
            cell_type=self.cell_type,
            self_loops_only=self.self_loops_only,
            rand_seed=self.rand_seed,
        )
        self.data_train = self.dataset.get_pygdata_obj_train(
            shuffle_seed=epoch_shuffle_seed(self.rand_seed, 0),
        )

        self.data_valtest = self.dataset.get_pygdata_obj_valtest()

    def current_epoch(self):
        """Epoch used to seed the training permutation; 0 when running standalone."""
        trainer = getattr(self, "trainer", None)
        return 0 if trainer is None else trainer.current_epoch

    def train_dataloader(self):
        # Lightning re-invokes this each epoch (reload_dataloaders_every_n_epochs=1), so
        # this is where the fresh permutation is drawn. Keyed on the epoch rather than on
        # a counter, so a repeated call within one epoch is a no-op rather than a redraw.
        epoch = self.current_epoch()
        self.dataset.reshuffle_train(self.data_train, epoch_shuffle_seed(self.rand_seed, epoch))

        og = NeighborLoader(
            self.data_train,
            input_nodes=self.data_train.train_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )
        return NeighborLoaderMod(og, self.n_hops)

    def val_dataloader(self):
        og = NeighborLoader(
            self.data_valtest,
            input_nodes=self.data_valtest.val_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )
        return NeighborLoaderMod(og, self.n_hops)

    def test_dataloader(self):
        og = NeighborLoader(
            self.data_valtest,
            input_nodes=self.data_valtest.test_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )
        return NeighborLoaderMod(og, self.n_hops)


def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2**32
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)
