"""Tests that temperature / noise parameters control mask sharpness.

Gumbel: tau controls how peaked the softmax samples are.
STG: sigma controls how noisy the stochastic gates are.
"""

import pytest
import torch

from gfs.models.feature_selection import GumbelFeatureSelector, STGFeatureSelector


@pytest.mark.slow
def test_gumbel_high_tau_more_uniform():
    """High tau produces more uniform masks; low tau produces peaked masks."""
    torch.manual_seed(42)
    selector = GumbelFeatureSelector(n_genes=50, n_select=5)
    selector.train()
    subgraph_id = torch.zeros(1, dtype=torch.long)

    # Sample masks at high tau
    high_tau_masks = []
    for _ in range(200):
        mask = selector.get_mask(tau=5.0, subgraph_id=subgraph_id)
        high_tau_masks.append(mask.detach())
    high_tau_mean = torch.stack(high_tau_masks).mean(dim=0).squeeze()

    # Sample masks at low tau
    low_tau_masks = []
    for _ in range(200):
        mask = selector.get_mask(tau=0.01, subgraph_id=subgraph_id)
        low_tau_masks.append(mask.detach())
    low_tau_mean = torch.stack(low_tau_masks).mean(dim=0).squeeze()

    assert high_tau_mean.max().item() < low_tau_mean.max().item(), (
        f"High tau max ({high_tau_mean.max():.4f}) should be less than "
        f"low tau max ({low_tau_mean.max():.4f})"
    )


@pytest.mark.slow
def test_stg_sigma_controls_sharpness():
    """High sigma produces more uniform mean masks; low sigma preserves mu structure."""
    torch.manual_seed(42)
    subgraph_id = torch.zeros(1, dtype=torch.long)

    # Use spread-out mu values so low sigma preserves structure, high sigma washes it out
    mu_init = torch.linspace(-2.0, 2.0, 50)

    # High sigma selector
    high_sigma = STGFeatureSelector(n_genes=50, n_select=5, sigma=2.0)
    high_sigma.train()
    with torch.no_grad():
        high_sigma.mu.copy_(mu_init)

    low_sigma = STGFeatureSelector(n_genes=50, n_select=5, sigma=0.1)
    low_sigma.train()
    with torch.no_grad():
        low_sigma.mu.copy_(mu_init)

    # Sample masks from high sigma
    high_masks = []
    for _ in range(200):
        mask = high_sigma.get_mask(subgraph_id=subgraph_id)
        high_masks.append(mask.detach())
    high_mean = torch.stack(high_masks).mean(dim=0).squeeze()

    # Sample masks from low sigma
    low_masks = []
    for _ in range(200):
        mask = low_sigma.get_mask(subgraph_id=subgraph_id)
        low_masks.append(mask.detach())
    low_mean = torch.stack(low_masks).mean(dim=0).squeeze()

    # With spread-out mu, low sigma preserves the structure (high std in mean mask)
    # High sigma noise washes out the structure, pushing mean mask toward 0.5 (lower std)
    assert high_mean.std().item() < low_mean.std().item(), (
        f"High sigma std ({high_mean.std():.4f}) should be less than "
        f"low sigma std ({low_mean.std():.4f}) — high noise washes out mu structure"
    )


@pytest.mark.slow
def test_gumbel_tau_convergence():
    """At very low tau, Gumbel mask should be nearly binary even during training."""
    torch.manual_seed(42)
    selector = GumbelFeatureSelector(n_genes=50, n_select=5)
    selector.train()
    subgraph_id = torch.zeros(1, dtype=torch.long)

    mask = selector.get_mask(tau=0.001, subgraph_id=subgraph_id).detach().squeeze()
    near_binary = ((mask < 0.05) | (mask > 0.95)).float().mean().item()

    assert near_binary > 0.9, (
        f"At tau=0.001, {near_binary:.1%} of mask values are near-binary "
        f"(expected >90%)"
    )


@pytest.mark.slow
def test_stg_gate_values_bounded():
    """STG gates should always be in [0, 1] due to hard_sigmoid clamping."""
    torch.manual_seed(42)
    selector = STGFeatureSelector(n_genes=50, n_select=5, sigma=1.0)
    selector.train()
    subgraph_id = torch.zeros(1, dtype=torch.long)

    for _ in range(100):
        mask = selector.get_mask(subgraph_id=subgraph_id).detach()
        assert mask.min().item() >= 0.0, f"Gate value below 0: {mask.min().item()}"
        assert mask.max().item() <= 1.0, f"Gate value above 1: {mask.max().item()}"
