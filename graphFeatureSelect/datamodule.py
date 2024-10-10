from pathlib import Path

import anndata as ad
import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from graphFeatureSelect.dataset import AnnDataGraphDataset
from graphFeatureSelect.utils import get_paths


class AnnDataGraphDataModule(L.LightningDataModule):
    def __init__(self, data_dir: None, file_names: list[str] = ["VISp_nhood.h5ad"], batch_size: int = 1):
        super().__init__()
        if data_dir is None:
            data_dir = get_paths()["data_root"]
            # data_dir = "../data/"
        self.adata_paths = [str(data_dir) + file_name for file_name in file_names]
        for adata_path in self.adata_paths:
            if not Path(adata_path).exists():
                raise FileNotFoundError(f"File not found: {adata_path}")

        self.batch_size = batch_size


    def setup(self, stage: str):
        self.adatas = []
        for adata_path in self.adata_paths:
            self.adatas.append(AnnDataGraphDataset(adata_path))
        self.data_full = ConcatDataset(self.adatas)
        self.data_train, self.data_test = random_split(self.data_full, [0.8, 0.2], generator=torch.Generator().manual_seed(0))

        if stage == "fit":
            self.data_train, self.data_val = random_split(self.data_train, [0.8, 0.2], generator=torch.Generator().manual_seed(1))

        if stage == "test": # Note: this is not the test set. Just a quick way to check the model through lightining.
            _, self.data_test = random_split(self.data_full, [0.9, 0.1], generator=torch.Generator().manual_seed(0))
            
        if stage == "predict":
            self.data_predict = self.data_full

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)
