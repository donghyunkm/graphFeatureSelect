import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from gfs.models.components import FeatureRegularizer


class GumbelFeatureSelector(nn.Module):
    """Gumbel-softmax based feature selection (persist method)."""

    def __init__(self, n_select, gene_ch):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(n_select, gene_ch))

    def forward(self, x, tau, subgraph_id):
        mask = self.get_mask(tau, subgraph_id)
        return mask * x

    def get_mask(self, tau, subgraph_id):
        if self.training:
            n_subgraphs = subgraph_id.max() + 1
            sample = F.gumbel_softmax(
                self.logits.unsqueeze(1).repeat(1, n_subgraphs, 1),
                tau=tau, hard=False, dim=-1,
            )
            k_hot = torch.max(sample, dim=0).values
            return k_hot[subgraph_id]
        else:
            k_hot_ind = torch.argmax(self.logits, dim=1)
            k_hot = torch.zeros(1, self.logits.size(1), device=self.logits.device)
            k_hot.scatter_(1, k_hot_ind.unsqueeze(0), 1)
            return k_hot

    def get_mask_indices(self):
        probs = F.softmax(self.logits, dim=1)
        mask_indices = torch.argmax(probs, dim=1)
        mask_probs = torch.gather(probs, dim=1, index=mask_indices.unsqueeze(1))
        return mask_indices, mask_probs.squeeze()

    def regularization_loss(self):
        return torch.tensor(0.0, device=self.logits.device)

    def on_train_epoch_end(self, logger_root_dir, current_epoch):
        mask_indices, mask_probs = self.get_mask_indices()
        metrics = {"epoch": current_epoch}

        for i in range(mask_indices.size(0)):
            metrics[f"sel_{i}"] = mask_indices[i].item()

        for i in range(mask_indices.size(0)):
            metrics[f"prob_{i}"] = mask_probs[i].item()

        path = logger_root_dir + "/selections.csv"
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(metrics.keys())
            writer.writerow([v for v in metrics.values()])


class ScGistFeatureSelector(nn.Module):
    """scGist-style feature selection with continuous logits."""

    def __init__(self, gene_ch, n_select):
        super().__init__()
        self.logits = nn.Parameter(torch.full((1, gene_ch), 0.5))
        self.feature_regularizer = FeatureRegularizer(l1=0.1, panel_size=n_select)

    def forward(self, x, tau=None, subgraph_id=None):
        return x * self.logits

    def regularization_loss(self):
        return self.feature_regularizer(self.logits) * 100

    def on_train_epoch_end(self, logger_root_dir, current_epoch):
        pass
