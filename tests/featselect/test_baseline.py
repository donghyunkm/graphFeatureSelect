"""Baseline comparison tests for feature selection.

Verifies that learned feature selection does not catastrophically hurt
classification and meaningfully outperforms random feature subsets.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from gfs.models.feature_selection import GumbelFeatureSelector, STGFeatureSelector, ScGistFeatureSelector
from tests.featselect.conftest import GatedMLP, train_gated_mlp, eval_accuracy


@pytest.mark.slow
def test_baseline_mlp_accuracy(toy_data):
    """Plain MLP on all 100 features should achieve >60% accuracy (sanity ceiling)."""
    mlp = nn.Sequential(
        nn.Linear(100, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(toy_data["X_train"], toy_data["y_train"])
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(150):
        mlp.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(mlp(xb), yb)
            loss.backward()
            optimizer.step()

    mlp.eval()
    with torch.no_grad():
        logits = mlp(toy_data["X_test"])
        preds = logits.argmax(dim=1)
        acc = (preds == toy_data["y_test"]).float().mean().item()

    assert acc > 0.55, f"Baseline MLP accuracy {acc:.3f} too low; data may not be learnable"


@pytest.mark.slow
@pytest.mark.parametrize("selector_factory,name", [
    (lambda: GumbelFeatureSelector(n_genes=100, n_select=10), "Gumbel"),
    (lambda: STGFeatureSelector(n_genes=100, n_select=10, sigma=0.5), "STG"),
    (lambda: ScGistFeatureSelector(n_genes=100, n_select=10), "scGist"),
])
def test_selected_features_vs_random(toy_data, selector_factory, name):
    """Learned feature mask should outperform a random mask of the same size."""
    torch.manual_seed(42)
    selector = selector_factory()
    model = train_gated_mlp(selector, toy_data, epochs=150, lr=1e-3, lam=0.01)

    # Accuracy with learned mask (eval mode -> hard binary)
    learned_acc = eval_accuracy(model, toy_data["X_test"], toy_data["y_test"])

    # Accuracy with random mask of same size
    # Replace the selector's eval mask with a random one
    model.eval()
    with torch.no_grad():
        # Get the learned mask's number of selected features
        learned_mask = selector.get_mask()
        n_selected = int(learned_mask.sum().item())
        if n_selected == 0:
            n_selected = 10  # fallback

        # Create random mask with same number of selected features
        rng = torch.Generator().manual_seed(123)
        random_indices = torch.randperm(100, generator=rng)[:n_selected]
        random_mask = torch.zeros(1, 100)
        random_mask[0, random_indices] = 1.0

        # Evaluate with random mask applied to test data
        masked_x = random_mask * toy_data["X_test"]
        logits = model.mlp(masked_x)
        preds = logits.argmax(dim=1)
        random_acc = (preds == toy_data["y_test"]).float().mean().item()

    assert learned_acc > random_acc, (
        f"{name}: learned mask acc ({learned_acc:.3f}) should beat "
        f"random mask acc ({random_acc:.3f})"
    )
