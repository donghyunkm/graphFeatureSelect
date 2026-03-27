import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FeatureSelector


class GumbelFeatureSelector(FeatureSelector):
    """Gumbel-softmax based feature selection (persist method).

    Learns n_select logit vectors over n_genes. At train time, each slot
    draws an independent Gumbel-softmax sample per subgraph, yielding a
    soft k-hot mask that is uniform within each subgraph. At eval time,
    hard argmax produces a deterministic binary mask.
    """

    def __init__(self, n_genes: int, n_select: int):
        super().__init__(n_genes, n_select)
        self.logits = nn.Parameter(torch.randn(n_select, n_genes))

    def forward(
        self, x: torch.Tensor, tau: float, subgraph_id: torch.Tensor
    ) -> torch.Tensor:
        mask = self.get_mask(tau, subgraph_id)
        return mask * x

    def get_mask(
        self, tau: float = 1.0, subgraph_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.training:
            n_subgraphs = subgraph_id.max().item() + 1
            # One Gumbel sample per subgraph per slot
            sample = F.gumbel_softmax(
                self.logits.unsqueeze(1).expand(-1, n_subgraphs, -1),
                tau=tau,
                hard=False,
                dim=-1,
            )
            # Max across slots -> soft k-hot per subgraph
            k_hot = sample.max(dim=0).values  # (n_subgraphs, n_genes)
            return k_hot[subgraph_id]  # (n_nodes, n_genes)
        else:
            # Hard argmax -> binary k-hot
            indices = self.logits.argmax(dim=1)
            mask = torch.zeros(self.n_genes, device=self.logits.device)
            mask.scatter_(0, indices, 1.0)
            return mask.unsqueeze(0)  # (1, n_genes) broadcasts over nodes

    def get_mask_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (selected gene indices, selection probabilities) per slot."""
        probs = F.softmax(self.logits, dim=1)
        mask_indices = torch.argmax(probs, dim=1)
        mask_probs = torch.gather(probs, dim=1, index=mask_indices.unsqueeze(1))
        return mask_indices, mask_probs.squeeze()


