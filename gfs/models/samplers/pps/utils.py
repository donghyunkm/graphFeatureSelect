import torch
from torch import Tensor


def clip_prob(p: Tensor) -> Tensor:
    """Clip probabilities to (0, 1) range."""
    eps = torch.finfo(p.dtype).eps
    p = torch.clip(p, eps, 1 - eps)
    return p


def deterministic_topk(p: Tensor, k: int) -> Tensor:
    """Return a mask of the top-k elements in p."""
    if isinstance(k, int):
        _, indices = torch.topk(p, k=k, dim=-1)
        mask = torch.zeros_like(p)
        mask.scatter_(-1, indices, 1)
        return mask
    elif isinstance(k, Tensor):
        sorted = torch.sort(p, dim=-1, descending=True).indices
        positions = torch.arange((p.shape[-1]), device=p.device).expand_as(p)
        mask = (positions < k).float()
        mask = torch.zeros_like(p).scatter_(-1, sorted, mask)
        return mask


def implicit_k(theta: Tensor) -> Tensor:
    """Compute the implcit k from logits."""
    return theta.sigmoid().sum(dim=-1, keepdim=True)
