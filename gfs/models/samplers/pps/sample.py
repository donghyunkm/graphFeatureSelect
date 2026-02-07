from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor

from .utils import clip_prob


def phantom_unit(func):

    def wrapper(p, *args, **kwargs):
        k = p.sum(-1, keepdim=True)
        phantom_prob = (1 - (k - k.floor())).clip(1e-6, 1 - 1e-6)
        p = torch.cat([p, phantom_prob], -1)  # Add phantom unit
        x = func(p, *args, **kwargs)
        x = x[..., :-1]  # Remove phantom unit
        return x

    return wrapper


@torch.no_grad()
def brewers_method(p: Tensor) -> Tensor:
    shape = p.shape
    if p.ndim == 1:
        p = p.unsqueeze(0)
    elif p.ndim >= 2:
        p = p.flatten(0, -2)

    k = p.sum(-1, keepdim=True).round().int()
    x = torch.zeros_like(p)
    p = clip_prob(p)
    for i in range(1, int(k.max().round()) + 1):
        c = k - (p * x).sum(dim=-1, keepdim=True)
        probs = (1 - x) * p * (c - p) / (c - p * (k - i + 1))
        probs = probs / probs.sum(dim=-1, keepdim=True)
        probs = clip_prob(probs)
        categorical = torch.multinomial(probs, num_samples=1, replacement=False)
        x = torch.where(i <= k, x.scatter(-1, categorical, 1), x)
    return x.reshape(shape)


def topk(x: torch.Tensor, k: int | torch.Tensor) -> torch.Tensor:
    max_k = int(k.max().item())
    idx = torch.topk(x, max_k, dim=-1).indices
    row_positions = torch.arange(max_k, device=x.device).expand(k.size(0), max_k)
    mask = (row_positions < k).float()
    return torch.zeros_like(x).scatter_(-1, idx, mask)


def pareto(p: Tensor, adjusted: bool = True) -> Tensor:
    """Pareto sampling (Rosén, 1997)"""
    k = p.sum(dim=-1, keepdim=True).round().int()
    eps = torch.finfo(p.dtype).eps
    p = p.clip(eps, 1 - eps)
    u = torch.rand_like(p).clip(eps, 1 - eps)
    q = (u / (1 - u)) / (p / (1 - p))
    if adjusted:
        d = (p * (1 - p)).sum(dim=-1, keepdim=True)
        q = q * torch.exp(p * (1 - p) * (p - 0.5) / d**2)
    x = topk(-q, k)
    return x


def sample_approx_khot(p: Tensor, k: Tensor, mask: Tensor) -> Tensor:
    """
    Helper function for iterative Poisson sampling

    Args:
        p (Tensor): Desired inclusion probabilities
        k (Tensor): Desired samples sizes
        mask (Tensor): Mask to apply to the probabilities
    """
    s = (p * mask).sum(-1, keepdim=True)
    n = mask.sum(-1, keepdim=True)
    q = torch.where(
        k <= s, k * p / s.clip(min=1), (n - k) * (1 - p) / (n - s).clip(min=1)
    )
    q = clip_prob(q)
    x = torch.bernoulli(q)
    x = torch.where(k <= s, x * mask, (1 - x) * mask)
    return x


@torch.no_grad()
def iterative_poisson_sampling(
    p: Tensor, k: Optional[Tensor], num_iter: int, num_samples: int = 1
) -> Tensor:
    """
    Iterative Poisson sampling (Pervez et al. 2023)

    Args:
        p (Tensor): Desired inclusion probabilities
        k (Tensor): Desired samples sizes
        num_iter (int): Number of iterations
    """
    if k is None:
        # If k is None, use the expected sample size rounded to the nearest integer
        k = p.sum(-1, keepdim=True).round()
    k = torch.ones_like(p) * k
    p = repeat(p, "b ... -> (n b) ...", n=num_samples)
    k = repeat(k, "b ... -> (n b) ...", n=num_samples)
    f = torch.ones_like(p)
    mask = torch.ones_like(p)
    pin = p.clone()
    kin = k

    x = torch.zeros_like(p)
    for _ in range(num_iter):
        sample = sample_approx_khot(p, k, mask)
        x = (x + f * sample).int()
        r = x.sum(-1, keepdim=True)
        leq = kin <= r
        f = torch.where(leq, -1, 1)
        mask = torch.where(leq, x, 1 - x)
        p = torch.where(leq, 1 - pin, pin)
        k = torch.where(leq, r - kin, kin - r)
    x = rearrange(x, "(n b) ... -> n b ...", n=num_samples)
    return x


def relaxed_topk(x, k, activation, hard=True):
    eps = torch.finfo(x.dtype).eps
    u = torch.rand_like(x).clip(eps, 1 - eps)
    g = -torch.log(-torch.log(u))
    x = activation(x + g, k)
    if hard:
        x_hard = topk(x, k)
        return x_hard + x - x.detach()
    else:
        return x


def gumbel_topk(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    u = torch.rand_like(x).clip(eps, 1 - eps)
    g = -torch.log(-torch.log(u))
    return topk(x + g, k)
    return topk(x + g, k)
