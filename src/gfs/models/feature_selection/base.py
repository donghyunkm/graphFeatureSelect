from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FeatureSelector(ABC, nn.Module):
    """Base class for feature selection layers.

    All selectors apply a mask to input gene expression:
    - Train: soft/stochastic mask for gradient flow
    - Eval: hard binary mask (constraint 1 from design)
    - Uniform mask within subgraphs (constraint 2)
    """

    def __init__(self, n_genes: int, n_select: int):
        super().__init__()
        self.n_genes = n_genes
        self.n_select = n_select

    @abstractmethod
    def forward(
        self, x: torch.Tensor, tau: float, subgraph_id: torch.Tensor
    ) -> torch.Tensor:
        """Apply feature mask to input expression.

        Args:
            x: (n_nodes, n_genes) gene expression
            tau: temperature (used by Gumbel, ignored by others)
            subgraph_id: (n_nodes,) subgraph assignment per node
        Returns:
            masked_x: (n_nodes, n_genes) masked expression
        """

    @abstractmethod
    def get_mask(
        self, tau: float = 1.0, subgraph_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return current feature mask.

        At eval: always binary (n_genes,) or (1, n_genes).
        At train: may be soft, shape depends on subgraph_id.
        """

    def regularization_loss(self) -> torch.Tensor:
        """Regularization term for feature selection params. Default: 0."""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def selected_indices(self) -> torch.Tensor:
        """Return indices of selected genes at eval time."""
        self.eval()
        with torch.no_grad():
            mask = self.get_mask()
        # Flatten to 1D if needed
        if mask.dim() > 1:
            mask = mask.squeeze(0)
        return torch.where(mask > 0.5)[0]
