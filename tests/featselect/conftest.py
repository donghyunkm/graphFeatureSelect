import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class GatedMLP(nn.Module):
    """Test harness: feature selector + MLP classifier."""
    def __init__(self, selector, n_features, n_classes):
        super().__init__()
        self.selector = selector
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        subgraph_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        masked = self.selector(x, tau=0.1, subgraph_id=subgraph_id)
        return self.mlp(masked)


def train_gated_mlp(selector, data, epochs=150, lr=1e-3, lam=0.0):
    """Train a GatedMLP and return the trained model."""
    model = GatedMLP(selector, n_features=100, n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(data["X_train"], data["y_train"])
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb) + lam * selector.regularization_loss()
            loss.backward()
            optimizer.step()

    return model


def eval_accuracy(model, X_test, y_test):
    """Evaluate classification accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        return (preds == y_test).float().mean().item()


@pytest.fixture(scope="module")
def toy_data():
    """Synthetic classification data with known informative features."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=5000, n_features=100, n_informative=10,
        n_redundant=0, n_repeated=0, n_classes=10,
        n_clusters_per_class=1, shuffle=False,
        class_sep=1.0, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.long),
        "informative": set(range(10)),
    }
