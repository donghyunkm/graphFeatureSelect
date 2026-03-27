"""Test that feature selectors recover truly informative features from synthetic data.

Each selector is paired with a small MLP and trained on a 10-class classification
problem where only 10 out of 100 features carry signal. After training, we check
that at least half the selected features are from the informative set.
"""

import pytest
import torch

from gfs.models.feature_selection.gumbel import GumbelFeatureSelector
from gfs.models.feature_selection.scgist import ScGistFeatureSelector
from gfs.models.feature_selection.stg import STGFeatureSelector

from tests.featselect.conftest import train_gated_mlp


@pytest.mark.slow
def test_gumbel_recovers_informative_features(toy_data):
    torch.manual_seed(42)
    selector = GumbelFeatureSelector(n_genes=100, n_select=10)
    # Gumbel needs more epochs: max-across-slots can have collisions early on
    train_gated_mlp(selector, toy_data, epochs=300, lr=2e-3, lam=0.0)

    selected = set(selector.selected_indices().tolist())
    overlap = selected & toy_data["informative"]
    assert len(overlap) >= 5, (
        f"Gumbel recovered only {len(overlap)}/10 informative features. "
        f"Selected: {sorted(selected)}, Expected subset of: {sorted(toy_data['informative'])}"
    )


@pytest.mark.slow
def test_stg_recovers_informative_features(toy_data):
    torch.manual_seed(42)
    selector = STGFeatureSelector(n_genes=100, n_select=10, sigma=0.5)
    train_gated_mlp(selector, toy_data, epochs=150, lr=1e-3, lam=0.1)

    selected = set(selector.selected_indices().tolist())
    overlap = selected & toy_data["informative"]
    assert len(overlap) >= 5, (
        f"STG recovered only {len(overlap)}/10 informative features. "
        f"Selected: {sorted(selected)}, Expected subset of: {sorted(toy_data['informative'])}"
    )


@pytest.mark.slow
def test_scgist_recovers_informative_features(toy_data):
    torch.manual_seed(42)
    selector = ScGistFeatureSelector(n_genes=100, n_select=10, l1=0.1)
    train_gated_mlp(selector, toy_data, epochs=150, lr=1e-3, lam=0.1)

    selected = set(selector.selected_indices().tolist())
    overlap = selected & toy_data["informative"]
    assert len(overlap) >= 5, (
        f"ScGist recovered only {len(overlap)}/10 informative features. "
        f"Selected: {sorted(selected)}, Expected subset of: {sorted(toy_data['informative'])}"
    )
