import torch
import torch.nn as nn

from .base import FeatureSelector


class ScGistFeatureSelector(FeatureSelector):
    """scGist-style feature selection with continuous logits.

    Learns a single weight vector over genes. At train time the continuous
    weights multiply the input (soft mask). At eval time, hard top-k
    binarization selects exactly n_select genes (constraint 1).

    Regularization pushes weights toward 0 or 1 and penalizes deviation
    from the target panel size.
    """

    def __init__(self, n_genes: int, n_select: int, l1: float = 0.1):
        super().__init__(n_genes, n_select)
        self.logits = nn.Parameter(torch.full((1, n_genes), 0.5))
        self.l1 = l1

    def forward(
        self, x: torch.Tensor, tau: float | None = None, subgraph_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = self.get_mask(tau, subgraph_id)
        return mask * x

    def get_mask(self, tau: float | None = None, subgraph_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.training:
            return self.logits  # continuous (1, n_genes), broadcasts
        else:
            # Hard top-k binary mask
            _, top_k_idx = self.logits.abs().topk(self.n_select, dim=1)
            mask = torch.zeros_like(self.logits)
            mask.scatter_(1, top_k_idx, 1.0)
            return mask  # (1, n_genes)

    def regularization_loss(self) -> torch.Tensor:
        abs_w = self.logits.abs()
        # Push weights toward 0 or 1
        binary_reg = (abs_w * (abs_w - 1).abs()).sum()
        # Panel size constraint
        size_reg = (abs_w.sum() - self.n_select).abs()
        return self.l1 * (binary_reg + size_reg)
