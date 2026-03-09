from gfs.models.samplers.gumbel_scheme import GumbelSampler
from gfs.models.samplers.imle_scheme import IMLESampler
from gfs.models.samplers.pps.pps import PPSSampler
from gfs.models.samplers.sfess.sfess import SFESSSampler
from gfs.models.samplers.simple_scheme import SIMPLESampler


class SamplerArgs:
    def __init__(
        self,
        *,
        name,
        sample_k,
        n_samples=1,
        assign_value=False,
        noise_scale=1.0,
        beta=1.0,
        tau=1.0,
        hard=True,
        pps_sample="pareto",
        pps_activation="sigmoid_topk",
        pps_gradient="straight_through",
    ):
        self.name = name
        self.sample_k = sample_k
        self.n_samples = n_samples
        self.assign_value = assign_value
        self.noise_scale = noise_scale
        self.beta = beta
        self.tau = tau
        self.hard = hard
        self.pps_sample = pps_sample
        self.pps_activation = pps_activation
        self.pps_gradient = pps_gradient


def get_sampler(sampler_args, device):
    if sampler_args is None:
        return None

    if sampler_args.name == "simple":
        return SIMPLESampler(
            sampler_args.sample_k,
            device=device,
            n_samples=sampler_args.n_samples,
            assign_value=sampler_args.assign_value,
        )
    elif sampler_args.name == "imle":
        return IMLESampler(
            sample_k=sampler_args.sample_k,
            device=device,
            n_samples=sampler_args.n_samples,
            noise_scale=sampler_args.noise_scale,
            beta=sampler_args.beta,
        )
    elif sampler_args.name == "gumbel":
        return GumbelSampler(
            k=sampler_args.sample_k,
            n_samples=sampler_args.n_samples,
            tau=sampler_args.tau,
            hard=sampler_args.hard,
        )
    elif sampler_args.name == "sfess":
        return SFESSSampler(
            k=sampler_args.sample_k,
            device=device,
            n_samples=sampler_args.n_samples,
        )
    elif sampler_args.name == "pps":
        return PPSSampler(
            k=sampler_args.sample_k,
            device=device,
            n_samples=sampler_args.n_samples,
            sampling=sampler_args.pps_sample,
            activation=sampler_args.pps_activation,
            gradient=sampler_args.pps_gradient,
        )
    else:
        raise ValueError(f"Unexpected sampler {sampler_args.name}")
