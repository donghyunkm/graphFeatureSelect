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
import random
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
            # make a 'scipy.sparse._csr.csr_matrix' all zeros
            adj = csr_matrix(np.zeros((self.adata.shape[0], self.adata.shape[0]), dtype=int))
            adj.setdiag(1)
            self.adj = adj
        else:
            adj = self.adata.obsp["spatial_connectivities"].copy()
            adj = adj.astype(bool).astype(int)
            adj.setdiag(0)  # removing self-loops
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



        # reproducible train/val split for crossvalidation 5 folds using StratifiedKFold
        self.cv = cv
        self.n_splits = n_splits
        assert self.cv < self.n_splits, "Crossvalidation index out of range"
        # skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        # splits = skf.split(self.adata, self.adata.obs[self.cell_type])
        # splits = list(splits)
        # self.train_ind = splits[self.cv][0]
        # self.val_ind = splits[self.cv][1]

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

        # boolean masks for train/val
        train_mask = torch.zeros(self.adata.shape[0], dtype=torch.bool)
        train_mask[self.train_ind] = True
        val_mask = torch.zeros(self.adata.shape[0], dtype=torch.bool)
        val_mask[self.val_ind] = True
        test_mask = torch.zeros(self.adata.shape[0], dtype=torch.bool)
        test_mask[self.test_ind] = True

        print("146 ", self.paths, self.paths[0])
        # if self.cv < 0: for normal model
        #     train_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec/train_mask.pt')
        #     val_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec/val_mask.pt')
        #     test_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec/test_mask.pt')

        if self.cv < 0: #for top10 subclass only
            print("CV < 0")
            # train_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec_top10/train_mask_top10.pt')
            # val_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec_top10/val_mask_top10.pt')
            # test_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec_top10/test_mask_top10.pt')
            train_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec_top10_1106/train_mask_top10.pt')
            val_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec_top10_1106/val_mask_top10.pt')
            test_mask = torch.load('/data/users1/dkim195/graphFeatureSelect/data/masks_one_sec_top10_1106/test_mask_top10.pt')

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
        # including self.dataset for debugging.
        # consider removing this if we run into cpu memory limits.
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

        self.dataset_test = PyGAnnData(
            self.test_paths,
            spatial_coords=self.spatial_coords,
            cell_type=self.cell_type,
            self_loops_only=self.self_loops_only,
            d_threshold=self.d_threshold,
            n_splits=self.n_splits,
            cv=self.cv,
            rand_seed=self.rand_seed,
            test_data=True
        )
        self.data_test = self.dataset_test.get_pygdata_obj()


    def train_dataloader(self):
        og = NeighborLoader(
            self.data,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32,
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
            num_workers=16,
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
            num_workers=16,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42)
        )
        return NeighborLoaderMod(og, self.n_hops)

    # def test_dataloader(self):
    #     og = NeighborLoader(
    #         self.data_test,
    #         input_nodes=self.data_test.val_mask,
    #         num_neighbors=[-1] * self.n_hops,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=16,
    #         worker_init_fn=seed_worker,
    #         generator=torch.Generator().manual_seed(42)
    #     )
    #     return NeighborLoaderMod(og, self.n_hops)

    def predict_dataloader(self):
        og = NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42)
        )
        return NeighborLoaderMod(og, self.n_hops)

def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2**32
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test_pyganndatagraphdatamodule():
    import numpy as np

    from gfs.data.hemisphere import PyGAnnDataGraphDataModule
    from gfs.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=path,
        file_names=["test_one_section_hemi.h5ad"],
        batch_size=2,
        n_hops=2,
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        d_threshold=1000,
        n_splits=5,
        cv=0,
        rand_seed=42,
    )
    datamodule.setup(stage="fit")

    # datamodule.data.train_mask

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


def numnodes_pyganndatagraphdatamodule_train():
    import numpy as np

    from gfs.data.hemisphere import PyGAnnDataGraphDataModule
    from gfs.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=path,
        file_names=["one_section_hemi_top10_1105.h5ad"],
        batch_size=2,
        n_hops=2,
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        d_threshold=1000,
        n_splits=5,
        cv=0,
        rand_seed=42,
    )
    datamodule.setup(stage="train")

    # datamodule.data.train_mask

    dataloader = iter(datamodule.train_dataloader())
    print("Num samples in train ", torch.sum(datamodule.data.train_mask)) # Num samples in train  tensor(46333)
    sum_nodes = 0
    for i, batch in enumerate(dataloader):
        sum_nodes += len(batch.input_id)

    print("sum input nodes: ", sum_nodes) # sum input nodes:  46333
    print("numnodes checks passed")

    return



def input_id_check():
    import numpy as np

    from gfs.data.hemisphere import PyGAnnDataGraphDataModule
    from gfs.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=path,
        file_names=["test_one_section_hemi.h5ad"],
        batch_size=2,
        n_hops=2,
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        d_threshold=1000,
        n_splits=5,
        cv=-1,
        rand_seed=42,
    )
    datamodule.setup(stage="train")

    # datamodule.data.train_mask

    dataloader = iter(datamodule.train_dataloader())
    print("Num samples in train ", torch.sum(datamodule.data.train_mask)) # Num samples in train  
    sum_nodes = 0
    for i, batch in enumerate(dataloader):
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        sum_nodes += len(idx)
    print("sum n_id nodes: ", sum_nodes) # sum n_id nodes:  46333
    print("numnodes checks passed")

    return





def numnodes_pyganndatagraphdatamodule():
    import numpy as np

    from gfs.data.hemisphere import PyGAnnDataGraphDataModule
    from gfs.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=path,
        file_names=["test_one_section_hemi.h5ad"],
        batch_size=2,
        n_hops=2,
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        d_threshold=1000,
        n_splits=5,
        cv=-1,
        rand_seed=42,
    )
    datamodule.setup(stage="train")

    # datamodule.data.train_mask

    dataloader = iter(datamodule.val_dataloader())
    print("Num samples in val ", torch.sum(datamodule.data.val_mask)) # Num samples in val  tensor(11584)
    sum_nodes = 0
    for i, batch in enumerate(dataloader):
        sum_nodes += len(batch.input_id)

    print("sum input nodes: ", sum_nodes) # sum input nodes:  11584
    print("numnodes checks passed")

    return

def numnodes_pyganndatagraphdatamodule2():
    import numpy as np

    from gfs.data.hemisphere import PyGAnnDataGraphDataModule
    from gfs.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=path,
        file_names=["test_one_section_hemi.h5ad"],
        batch_size=4, # seed/root nodes
        n_hops=2,
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        d_threshold=1000,
        n_splits=5,
        cv=-1,
        rand_seed=42,
    )
    datamodule.setup(stage="train")

    # datamodule.data.train_mask

    dataloader = iter(datamodule.val_dataloader())
    print("Num samples in val ", torch.sum(datamodule.data.val_mask)) # Num samples in val  tensor(11584)
    sum_nodes = 0
    for i, batch in enumerate(dataloader):
        sum_nodes += int(torch.sum(batch.val_mask))
 
    print("sum input nodes: ", sum_nodes) # sum all neighboring nodes and input nodes:  126495
    print("numnodes checks passed")

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
        d_threshold=1000,
        n_splits=5,
        cv=0,
        rand_seed=42,
    )
    print("pyganndata checks passed")

    return pygdata


def test_seed():
    from gfs.data.hemisphere import PyGAnnDataGraphDataModule
    from lightning.pytorch import seed_everything
    from gfs.utils import get_paths

    path = get_paths()["data_root"]

    for i in range(3):
        seed_everything(i, workers=True)
        datamodule = PyGAnnDataGraphDataModule(
            data_dir=path,
            file_names=["test_one_section_hemi.h5ad"],
            batch_size=2, # seed nodes
            n_hops=2,
            cell_type="subclass",
            spatial_coords=["x_section", "y_section", "z_section"],
            d_threshold=1000,
            n_splits=5,
            cv=0,
            rand_seed=42,
        )
        datamodule.setup(stage="fit")
        dataloader = iter(datamodule.train_dataloader())

        batch = next(dataloader)
        print(batch)

    # check if dataloader returns same batch for different rand seed 

    print("random seed checks passed")

    return

def mask_test():
    import numpy as np

    from gfs.data.hemisphere import PyGAnnDataGraphDataModule
    from gfs.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=path,
        file_names=["train.h5ad"],
        batch_size=2,
        n_hops=2,
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        d_threshold=1000,
        n_splits=5,
        cv=-1,
        rand_seed=42,
    )
    datamodule.setup(stage="train")

    # datamodule.data.train_mask

    dataloader = iter(datamodule.train_dataloader())
    sum_nodes = 0
    train_nodes = []
    for i, batch in enumerate(dataloader):
        train_nodes = train_nodes + batch.n_id[batch.train_mask].tolist()

    datamodule2 = PyGAnnDataGraphDataModule(
        data_dir=path,
        file_names=["train.h5ad"],
        batch_size=2,
        n_hops=2,
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        d_threshold=1000,
        n_splits=5,
        cv=-1,
        rand_seed=42,
    )
    datamodule2.setup(stage="val")

    # datamodule.data.train_mask

    dataloader2 = iter(datamodule2.train_dataloader())
    val_nodes = []
    for i, batch in enumerate(dataloader2):
        val_nodes = val_nodes + batch.n_id[batch.val_mask].tolist()
    
    
    set1 = set(train_nodes)
    set2 = set(val_nodes)

    # Check for intersection
    if set1.intersection(set2):
        print("Overlap exists.")
    else:
        print("No overlap.")
    return


if __name__ == "__main__":
    # print("running pyganndata")
    # test_pyganndata()
    # print("running pyganndata")
    # test_pyganndatagraphdatamodule()
    # print("testing pyganndata (random seed)")
    # test_seed()

    print("test")
    numnodes_pyganndatagraphdatamodule_train()