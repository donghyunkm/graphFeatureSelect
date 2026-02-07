from functools import partial
from typing import Callable, Optional

from torch import Tensor

from .gradient import reinmax, straight_through
from .projection import scaled_sigmoid, sequential_softmax
from .sample import (
    brewers_method,
    gumbel_topk,
    iterative_poisson_sampling,
    pareto,
    phantom_unit,
    relaxed_topk,
)
from .sigmoid_topk import sigmoid_topk
from .soft_topk import soft_topk


def sample_subset(
    activation: Callable,
    sample: Callable,
    gradient: Callable,
    theta: Tensor,
    k: Optional[Tensor] = None,
    num_samples: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Sample a subset proportionally to theta with subset size k.
    If k is None, the expected subset size is the inclusion probabilities' sum.
    The sample is differentiable with respect to theta.
    Returns sample and inclusion probabilities.
    """
    if gradient is reinmax:
        if k is None:
            raise ValueError("k must be provided when using ReinMax")
        x, p = reinmax(theta, k)
        return x, p
    if sample is None:
        if activation is sequential_softmax:
            activation_tau = partial(sequential_softmax, tau=1.0)
        else:
            activation_tau = lambda x, k: activation(x / 1.0, k)
        x = relaxed_topk(theta, k, activation_tau, hard=True)
        return x, None
    if sample is gumbel_topk:
        assert k is not None, "k must be provided for gumbel_topk"
        x = sample(theta, k)
        p = sequential_softmax(theta, k)
    elif k is None:
        p = theta.sigmoid()
        x = sample(p)
    else:
        p = activation(theta, k)
        x = sample(p)
    x = gradient(p, x)
    return x, p


def make_sampler(
    activation: str = "sigmoid_topk",
    sampling: str = "brewers_method",
    gradient: str = "reinmax",
    use_phantom_unit: bool = False,
) -> Callable:
    def sampler(
        theta: Tensor,
        k: Optional[Tensor] = None,
        num_samples: int = 1,
    ) -> tuple[Tensor, Tensor]:
        activation_fn = {
            "sigmoid_topk": sigmoid_topk,
            "scaled_sigmoid": scaled_sigmoid,
            "sequential_softmax": sequential_softmax,
            "soft_topk": soft_topk,
        }[activation]
        sampling_fn = {
            "brewers_method": brewers_method,
            "iterative_poisson_sampling": partial(
                iterative_poisson_sampling, k=k, num_iter=7
            ),
            "pareto": pareto,
            "relaxed": None,
            "gumbel_topk": gumbel_topk,
        }[sampling]
        if use_phantom_unit:
            sampling_fn = phantom_unit(sampling_fn)
        gradient_fn = {
            "reinmax": reinmax,
            "straight_through": straight_through,
        }[gradient]

        return sample_subset(
            activation=activation_fn,
            sample=sampling_fn,
            gradient=gradient_fn,
            theta=theta,
            k=k,
            num_samples=num_samples,
        )

    return sampler
    return sampler
    return sampler
