"""Integration tests for feature selection modules.

Uses realistic tensor shapes from the dev dataset (485 genes, ~200 nodes,
5 subgraphs) but does not load actual h5ad files.
"""

import pytest
import torch

from gfs.models.feature_selection.gumbel import GumbelFeatureSelector
from gfs.models.feature_selection.scgist import ScGistFeatureSelector
from gfs.models.feature_selection.stg import STGFeatureSelector

N_GENES = 485
N_SELECT = 10
N_NODES = 200
N_SUBGRAPHS = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def x():
    return torch.randn(N_NODES, N_GENES)


@pytest.fixture
def subgraph_id():
    return torch.repeat_interleave(torch.arange(N_SUBGRAPHS), N_NODES // N_SUBGRAPHS)


@pytest.fixture
def gumbel():
    return GumbelFeatureSelector(n_genes=N_GENES, n_select=N_SELECT)


@pytest.fixture
def stg():
    return STGFeatureSelector(n_genes=N_GENES, n_select=N_SELECT, sigma=0.5)


@pytest.fixture
def scgist():
    return ScGistFeatureSelector(n_genes=N_GENES, n_select=N_SELECT)


@pytest.fixture(params=["gumbel", "stg", "scgist"])
def selector(request, gumbel, stg, scgist):
    return {"gumbel": gumbel, "stg": stg, "scgist": scgist}[request.param]


# ---------------------------------------------------------------------------
# GumbelFeatureSelector
# ---------------------------------------------------------------------------

class TestGumbelFeatureSelector:
    def test_forward_shape(self, gumbel, x, subgraph_id):
        gumbel.train()
        out = gumbel(x, tau=1.0, subgraph_id=subgraph_id)
        assert out.shape == x.shape

    def test_eval_forward_shape(self, gumbel, x, subgraph_id):
        gumbel.eval()
        out = gumbel(x, tau=1.0, subgraph_id=subgraph_id)
        assert out.shape == x.shape

    def test_eval_mask_binary(self, gumbel, subgraph_id):
        gumbel.eval()
        mask = gumbel.get_mask(tau=1.0, subgraph_id=subgraph_id)
        assert ((mask == 0) | (mask == 1)).all()

    def test_eval_mask_selects_n(self, gumbel, subgraph_id):
        """At eval, at most n_select genes are selected (argmax can collapse)."""
        gumbel.eval()
        mask = gumbel.get_mask(tau=1.0, subgraph_id=subgraph_id)
        unique_mask = mask[0] if mask.dim() == 2 else mask
        n_selected = unique_mask.sum().item()
        assert 1 <= n_selected <= N_SELECT

    def test_train_mask_soft(self, gumbel, subgraph_id):
        gumbel.train()
        mask = gumbel.get_mask(tau=1.0, subgraph_id=subgraph_id)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0 + 1e-6
        intermediate = (mask > 0.01) & (mask < 0.99)
        assert intermediate.any(), "Expected some intermediate (soft) mask values during training"

    def test_uniform_within_subgraph(self, gumbel, subgraph_id):
        gumbel.train()
        mask = gumbel.get_mask(tau=1.0, subgraph_id=subgraph_id)
        for sg_id in subgraph_id.unique():
            sg_nodes = (subgraph_id == sg_id).nonzero().squeeze()
            masks_in_sg = mask[sg_nodes]
            assert (masks_in_sg == masks_in_sg[0]).all(), (
                f"Mask varies within subgraph {sg_id.item()}"
            )

    def test_different_subgraphs_can_differ(self, gumbel, subgraph_id):
        gumbel.train()
        mask = gumbel.get_mask(tau=1.0, subgraph_id=subgraph_id)
        first_nodes = [0, 40, 80, 120, 160]
        unique_masks = mask[first_nodes]
        assert len(unique_masks.unique(dim=0)) > 1, (
            "Expected different subgraphs to have different masks"
        )

    def test_regularization_loss(self, gumbel):
        loss = gumbel.regularization_loss()
        assert loss.shape == ()

    def test_get_mask_indices(self, gumbel):
        indices, probs = gumbel.get_mask_indices()
        assert indices.shape == (N_SELECT,)
        assert probs.shape == (N_SELECT,)
        assert indices.min() >= 0
        assert indices.max() < N_GENES

    def test_selected_indices(self, gumbel):
        gumbel.eval()
        indices = gumbel.selected_indices()
        assert indices.min() >= 0
        assert indices.max() < N_GENES
        assert len(indices.unique()) == len(indices)

    def test_gradient_flows(self, gumbel, x, subgraph_id):
        gumbel.train()
        out = gumbel(x, tau=1.0, subgraph_id=subgraph_id)
        loss = out.sum() + gumbel.regularization_loss()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in gumbel.parameters())
        assert has_grad, "No gradient flowed to any parameter"


# ---------------------------------------------------------------------------
# STGFeatureSelector
# ---------------------------------------------------------------------------

class TestSTGFeatureSelector:
    def test_forward_shape(self, stg, x, subgraph_id):
        stg.train()
        out = stg(x, tau=1.0, subgraph_id=subgraph_id)
        assert out.shape == x.shape

    def test_eval_forward_shape(self, stg, x, subgraph_id):
        stg.eval()
        out = stg(x, tau=1.0, subgraph_id=subgraph_id)
        assert out.shape == x.shape

    def test_eval_mask_binary(self, stg):
        """At eval, mask should be hard binary (0 or 1)."""
        stg.eval()
        mask = stg.get_mask()
        assert ((mask == 0) | (mask == 1)).all()

    def test_eval_mask_selects_n(self, stg):
        stg.eval()
        mask = stg.get_mask()
        unique_mask = mask[0] if mask.dim() == 2 else mask
        assert unique_mask.sum().item() == N_SELECT

    def test_train_mask_soft(self, stg, subgraph_id):
        stg.train()
        mask = stg.get_mask(tau=1.0, subgraph_id=subgraph_id)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_regularization_loss(self, stg):
        loss = stg.regularization_loss()
        assert loss.shape == ()
        assert loss.requires_grad

    def test_regularization_loss_in_valid_range(self, stg):
        """Reg loss is mean of gaussian CDF values, so in [0, 1]."""
        loss = stg.regularization_loss()
        assert 0.0 <= loss.item() <= 1.0

    def test_selected_indices(self, stg):
        stg.eval()
        indices = stg.selected_indices()
        assert len(indices) == N_SELECT
        assert indices.min() >= 0
        assert indices.max() < N_GENES
        assert len(indices.unique()) == len(indices)

    def test_gradient_flows(self, stg, x, subgraph_id):
        stg.train()
        out = stg(x, tau=1.0, subgraph_id=subgraph_id)
        loss = out.sum() + stg.regularization_loss()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in stg.parameters())
        assert has_grad, "No gradient flowed to any parameter"

    def test_eval_deterministic(self, stg, x, subgraph_id):
        """At eval, repeated calls should give the same result."""
        stg.eval()
        out1 = stg(x, tau=1.0, subgraph_id=subgraph_id)
        out2 = stg(x, tau=1.0, subgraph_id=subgraph_id)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# ScGistFeatureSelector
# ---------------------------------------------------------------------------

class TestScGistFeatureSelector:
    def test_forward_shape(self, scgist, x, subgraph_id):
        scgist.train()
        out = scgist(x, tau=1.0, subgraph_id=subgraph_id)
        assert out.shape == x.shape

    def test_eval_forward_shape(self, scgist, x, subgraph_id):
        scgist.eval()
        out = scgist(x, tau=1.0, subgraph_id=subgraph_id)
        assert out.shape == x.shape

    def test_eval_mask_binary(self, scgist):
        scgist.eval()
        mask = scgist.get_mask()
        assert ((mask == 0) | (mask == 1)).all()

    def test_eval_mask_selects_n(self, scgist):
        scgist.eval()
        mask = scgist.get_mask()
        unique_mask = mask[0] if mask.dim() == 2 else mask
        assert unique_mask.sum().item() == N_SELECT

    def test_deterministic(self, scgist, x, subgraph_id):
        """ScGist is deterministic -- no stochastic sampling."""
        scgist.train()
        out1 = scgist(x, tau=1.0, subgraph_id=subgraph_id)
        out2 = scgist(x, tau=1.0, subgraph_id=subgraph_id)
        assert torch.allclose(out1, out2)

    def test_regularization_loss(self, scgist):
        loss = scgist.regularization_loss()
        assert loss.shape == ()
        assert loss.requires_grad

    def test_regularization_loss_positive(self, scgist):
        """With default logits and panel_size=10, reg loss should be > 0."""
        loss = scgist.regularization_loss()
        assert loss.item() > 0.0

    def test_selected_indices(self, scgist):
        scgist.eval()
        indices = scgist.selected_indices()
        assert len(indices) == N_SELECT
        assert indices.min() >= 0
        assert indices.max() < N_GENES
        assert len(indices.unique()) == len(indices)

    def test_gradient_flows(self, scgist, x, subgraph_id):
        scgist.train()
        out = scgist(x, tau=1.0, subgraph_id=subgraph_id)
        loss = out.sum() + scgist.regularization_loss()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in scgist.parameters())
        assert has_grad, "No gradient flowed to any parameter"

    def test_regularization_encourages_sparsity(self, scgist):
        """Reg loss should decrease when logits are closer to a sparse binary mask."""
        loss_init = scgist.regularization_loss().item()
        with torch.no_grad():
            scgist.logits.zero_()
            scgist.logits[0, :N_SELECT] = 1.0
        loss_sparse = scgist.regularization_loss().item()
        assert loss_sparse < loss_init, (
            f"Sparse mask loss ({loss_sparse}) should be less than init loss ({loss_init})"
        )


# ---------------------------------------------------------------------------
# Cross-selector tests (parameterized)
# ---------------------------------------------------------------------------

class TestAllSelectors:
    def test_forward_returns_tensor(self, selector, x, subgraph_id):
        selector.train()
        out = selector(x, tau=1.0, subgraph_id=subgraph_id)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape

    def test_forward_eval_returns_tensor(self, selector, x, subgraph_id):
        selector.eval()
        out = selector(x, tau=1.0, subgraph_id=subgraph_id)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape

    def test_eval_mask_binary(self, selector):
        selector.eval()
        mask = selector.get_mask()
        assert ((mask == 0) | (mask == 1)).all(), "Eval mask must be strictly binary"

    def test_eval_mask_selects_n(self, selector):
        selector.eval()
        mask = selector.get_mask()
        unique_mask = mask[0] if mask.dim() == 2 else mask
        n_selected = unique_mask.sum().item()
        assert 1 <= n_selected <= N_SELECT

    def test_regularization_loss_is_scalar(self, selector):
        loss = selector.regularization_loss()
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_has_learnable_parameters(self, selector):
        params = list(selector.parameters())
        assert len(params) > 0
        total_params = sum(p.numel() for p in params)
        assert total_params > 0

    def test_gradient_flows_through_forward(self, selector, x, subgraph_id):
        selector.train()
        out = selector(x, tau=1.0, subgraph_id=subgraph_id)
        loss = out.sum() + selector.regularization_loss()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in selector.parameters())
        assert has_grad

    def test_output_dtype_matches_input(self, selector, x, subgraph_id):
        selector.train()
        out = selector(x, tau=1.0, subgraph_id=subgraph_id)
        assert out.dtype == x.dtype

    def test_selected_indices_valid(self, selector):
        selector.eval()
        indices = selector.selected_indices()
        assert indices.min() >= 0
        assert indices.max() < N_GENES
        assert len(indices.unique()) == len(indices), "selected_indices has duplicates"
