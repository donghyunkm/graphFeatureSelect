import math

import torch
import torch.nn as nn

from .base import FeatureSelector


class STGFeatureSelector(FeatureSelector):
    """Stochastic Gates feature selection.

    Train: hard_sigmoid(mu + sigma * noise), noise sampled per subgraph
    Eval: top-k genes by mu -> binary mask
    """

    def __init__(self, n_genes: int, n_select: int, sigma: float = 0.5):
        super().__init__(n_genes, n_select)
        self.mu = nn.Parameter(0.01 * torch.randn(n_genes))
        self.sigma = sigma

    def forward(self, x, tau=None, subgraph_id=None):
        mask = self.get_mask(tau, subgraph_id)
        return mask * x

    def get_mask(self, tau=None, subgraph_id=None):
        if self.training:
            if subgraph_id is not None:
                n_subgraphs = subgraph_id.max().item() + 1
                # Sample independent noise per subgraph
                noise = torch.randn(n_subgraphs, self.n_genes, device=self.mu.device)
                z = self.mu.unsqueeze(0) + self.sigma * noise  # (n_subgraphs, n_genes)
                gates = self._hard_sigmoid(z)
                return gates[subgraph_id]  # (n_nodes, n_genes)
            else:
                # Fallback: single noise sample
                noise = torch.randn_like(self.mu)
                z = self.mu + self.sigma * noise
                return self._hard_sigmoid(z).unsqueeze(0)
        else:
            # Hard top-k binary mask by mu values
            _, top_k_idx = self.mu.topk(self.n_select)
            mask = torch.zeros(self.n_genes, device=self.mu.device)
            mask.scatter_(0, top_k_idx, 1.0)
            return mask.unsqueeze(0)  # (1, n_genes)

    def _hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularization_loss(self):
        """Penalizes open gates via Gaussian CDF."""
        return torch.mean(self._gaussian_cdf((self.mu + 0.5) / self.sigma))

    def _gaussian_cdf(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
