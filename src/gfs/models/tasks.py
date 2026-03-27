"""Task heads for node-level prediction."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Linear classification head: embeddings → class logits.

    Args:
        in_ch: input embedding dimension (hid_ch from backbone)
        n_classes: number of output classes
    """

    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (n_nodes, n_classes)."""
        return self.linear(x)


class ReconstructionHead(nn.Module):
    """MLP reconstruction head: embeddings → predicted gene expression.

    Predicts the full (unmasked) expression profile from node embeddings.
    Used to validate that selected genes carry enough information.

    Args:
        in_ch: input embedding dimension
        n_genes: number of genes to reconstruct
        hidden: list of hidden layer widths
    """

    def __init__(self, in_ch: int, n_genes: int, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [128, 128]

        layers = []
        prev = in_ch
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_genes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted expression (n_nodes, n_genes)."""
        return self.mlp(x)
