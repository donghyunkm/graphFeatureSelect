import warnings
from functools import partial

import torch


def sigmoid_topk(
    x: torch.Tensor,
    k: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 50,
    use_float64: bool = True,
) -> torch.Tensor:
    """Sigmoid top-k activation that maps R^n to [0, 1]^n with sum k.
    It is differentiable with respect to both x and k and can be used as a
    continuous relaxation of top-k selection.

    Args:
        x (Tensor): Input tensor of shape (batch_size, n)
        k (Tensor): Target sum (batch_size, 1)
        tol (float, optional): Tolerance for the root-finding solver. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations for the solver.
            Defaults to 50.
        use_float64 (bool, optional): Whether to use float64 internally in the solver.
            This allows solving the root-finding problem more accurately at the cost of
            some speed and memory. Regardless of this setting, the output will be in
            the same dtype as the input. Defaults to True.

    Returns:
        Tensor: Output tensor of shape (batch_size, n) with values in [0, 1] and sum k.
    """
    assert (0 < k).all() and (k < x.size(-1)).all(), "Invalid subset size"
    solver = partial(halleys, tol=tol, max_iter=max_iter, use_float64=use_float64)
    return SigmoidTopKFunction.apply(x, k, solver)  # type: ignore


class SigmoidTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, solver):
        c = solver(x, k)
        p = (x + c).sigmoid().to(x.dtype)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        grad_p = p * (1 - p)
        sum_grad_p = grad_p.sum(dim=-1, keepdim=True)
        dp_dx = grad_p * grad_output
        dc_dx = -grad_p / sum_grad_p
        grad_x = dp_dx + dp_dx.sum(dim=-1, keepdim=True) * dc_dx
        dc_dk = 1 / sum_grad_p
        grad_k = (grad_p * dc_dk * grad_output).sum(dim=-1, keepdim=True)
        return grad_x, grad_k, None


def quantile(x, q):
    x_sorted = x.sort(dim=-1).values
    idx = (q * (x.size(-1) - 1)).floor().long()
    return torch.gather(x_sorted, -1, idx)


def halleys(x, k, tol, max_iter, use_float64):
    n = x.size(-1)

    bound = x.abs().max(dim=-1, keepdim=True).values + 10  # σ(10) ≈ 1
    if use_float64:
        bound = bound.to(torch.float64)
    low, high = -bound, bound

    c_logit = (k / n).logit() - x.mean(dim=-1, keepdim=True)
    s_logit = (x + c_logit).sigmoid()
    f_logit = s_logit.sum(dim=-1, keepdim=True) - k

    c_quantile = -quantile(x, (n - k) / n)
    s_quantile = (x + c_quantile).sigmoid()
    f_quantile = s_quantile.sum(dim=-1, keepdim=True) - k

    logit_better = f_logit.abs() < f_quantile.abs()
    c = torch.where(logit_better, c_logit, c_quantile)
    s = torch.where(logit_better, s_logit, s_quantile)
    f = torch.where(logit_better, f_logit, f_quantile)

    for _ in range(max_iter):
        df_dc = (s * (1 - s)).sum(dim=-1, keepdim=True)
        d2f_dc2 = ((s * (1 - s)) * (1 - 2 * s)).sum(dim=-1, keepdim=True)
        c_halley = c - 2 * f * df_dc / (2 * (df_dc**2) - f * d2f_dc2 + 1e-12)
        s_halley = torch.sigmoid(x + c_halley)
        f_halley = s_halley.sum(dim=-1, keepdim=True) - k

        c_bisection = 0.5 * (low + high)
        s_bisection = (x + c_bisection).sigmoid()
        f_bisection = s_bisection.sum(dim=-1, keepdim=True) - k

        halley_better = f_halley.abs() < f_bisection.abs()
        c = torch.where(halley_better, c_halley, c_bisection)
        s = torch.where(halley_better, s_halley, s_bisection)
        f = torch.where(halley_better, f_halley, f_bisection)

        if f.abs().max() < tol:
            return c

        high = torch.where(f_bisection < 0, high, c_bisection)
        low = torch.where(f_bisection > 0, low, c_bisection)
        high = torch.where(f < 0, high, c)
        low = torch.where(f > 0, low, c)

    warnings.warn(f"Halley's method did not converge", RuntimeWarning)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    b, n = 8, 64
    x = torch.randn(b, n, requires_grad=True, dtype=torch.float64)
    k = torch.rand(b, 1, requires_grad=True, dtype=torch.float64) * (n - 2) + 1
    torch.autograd.gradcheck(sigmoid_topk, (x, k))  # type: ignore
    print("Gradcheck passed!")
