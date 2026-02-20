import torch
import torch.nn as nn

from .subset import make_sampler


class PPSSampler(nn.Module):

    def __init__(
        self,
        k,
        device,
        n_samples=1,
        activation="sequential_softmax",
        sampling="gumbel_topk",
        gradient="straight_through",
    ):
        super(PPSSampler, self).__init__()
        self.k = torch.tensor(k, device=device)
        self.device = device
        assert n_samples > 0
        self.n_samples = n_samples
        self.sampler = make_sampler(activation, sampling, gradient)

    def forward(self, scores):
        batch, choices, _ = scores.shape
        flat_scores = scores.permute((0, 2, 1)).reshape(batch, choices)
        k = self.k.repeat(batch, 1)
        samples, _ = self.sampler(flat_scores.repeat(self.n_samples, 1), k)
        samples = samples.reshape(self.n_samples, batch, choices)
        return samples, None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = PPSSampler(2, device, n_samples=10)
    scores = torch.rand(64, 4, 1, device=device)
    samples, log_p = sampler(scores)
    print(samples.shape)
    print(samples.shape)
