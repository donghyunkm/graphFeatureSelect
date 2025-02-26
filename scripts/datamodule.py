# combines functionality from dataloaders.py and dataprocessors.py from nichecompass
import lightning as L
import torch
from torch_geometric.data import Data
from torch_geometric.loader.neighbor_loader import NeighborLoader
from torch_geometric.transforms import NodePropertySplit, RandomNodeSplit

from gfs.datasets import AnnDataGraphDataset
from gfs.utils import get_paths


class AnnDataGraphDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: None,
        file_names: list[str] = ["VISp_nhood.h5ad"],
        batch_size: int = 1,
        n_hops: int = 2,
        no_edges: bool = False,
    ):
        super().__init__()
        if data_dir is None:
            data_dir = get_paths()["data_root"]
            # data_dir = "../data/"
        self.adata_paths = [str(data_dir) + file_name for file_name in file_names]
        self.batch_size = batch_size
        self.n_hops = n_hops
        self.no_edges = no_edges

    def node_mask(self, method, data):
        if method == "rand":
            random_node_split = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.2, key="labels")
            data = random_node_split(data)
        else:
            node_property_split = NodePropertySplit(
                property_name="popularity", ratios=[0.6, 0.1, 0.1, 0.1, 0.1], ascending=True
            )
            data = node_property_split(data)

        return data

    def setup(self, stage=None):
        dataset = AnnDataGraphDataset(self.adata_paths)
        if self.no_edges:
            self_edges = torch.tensor([[i, i] for i in range(dataset.x.shape[0])]).T
            data = Data(x=dataset.x, edge_index=self_edges, labels=dataset.labels)
        else:
            data = Data(x=dataset.x, edge_index=dataset.edge_index, labels=dataset.labels)
        self.data = self.node_mask("rand", data)
        self.dataset = dataset

    def train_dataloader(self):
        return NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
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
