import math

import torch
import torch.nn as nn


class STGFeatureSelector(nn.Module):
    """Stochastic Gates (STG) feature selection."""

    def __init__(self, input_dim, sigma):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(input_dim), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma

    def forward(self, x, tau=None, subgraph_id=None):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        return x * stochastic_gate

    def get_mask(self):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        return stochastic_gate

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def _gaussian_cdf(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def regularization_loss(self):
        return torch.mean(self._gaussian_cdf((self.mu + 0.5) / self.sigma))

    def on_train_epoch_end(self, logger_root_dir, current_epoch):
        mask = self.get_mask()
        nonzero_elements = torch.count_nonzero(mask)
        print("Nonzero ", nonzero_elements)
        print("Mask ", mask)

    def _apply(self, fn):
        super()._apply(fn)
        self.noise = fn(self.noise)
        return self
