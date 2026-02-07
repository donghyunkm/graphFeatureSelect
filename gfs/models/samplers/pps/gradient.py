import torch
from torch import Tensor

from .sample import pareto
from .sigmoid_topk import sigmoid_topk


def straight_through(pi: Tensor, x: Tensor) -> Tensor:
    x = (x - pi).detach() + pi
    return x


def reinmax(theta: Tensor, k: Tensor):
    eps = torch.finfo(theta.dtype).eps

    # Find probabilities and sample
    pi = sigmoid_topk(theta, k)
    x = pareto(pi)

    # Perturb probabilities towards the sample
    pi_1 = 0.5 * (x + pi)
    theta_1 = pi_1.clip(eps, 1 - eps).logit()

    # Take the gradient of the original parameters at the perturbed parameters
    pi_1 = sigmoid_topk((theta_1 - theta).detach() + theta, k)

    pi_2 = 2.0 * pi_1 - 0.5 * pi
    x = (x - pi_2).detach() + pi_2
    return x, pi
