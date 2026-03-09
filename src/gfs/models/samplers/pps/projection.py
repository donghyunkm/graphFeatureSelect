import torch


def scaled_sigmoid(theta, k):
    """From "Neural Conditional Poisson Sampling" by Pervez et al. (2023)"""
    n = theta.size(-1)
    p = torch.sigmoid(theta)
    s = p.sum(dim=-1, keepdim=True)
    q = torch.where(k <= s, k * p / s, 1 - (n - k) * (1 - p) / (n - s))
    return q


def sequential_softmax(x, k, tau=1.0):
    """Sequential softmax by Xie and Ermon (2019)"""
    k = int(k[0][0].item())
    eps = torch.finfo(torch.float32).tiny
    p = torch.zeros_like(x)
    onehot_approx = torch.zeros_like(x)
    for _ in range(k):
        khot_mask = (1.0 - onehot_approx).clip(eps)
        x = x + torch.log(khot_mask)
        onehot_approx = torch.softmax(x / tau, dim=-1)
        p += onehot_approx
    return p
