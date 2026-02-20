import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.ops import sigmoid_focal_loss

from gfs.models.transforms import HalfHop
from gfs.models.get_sampler import SamplerArgs, get_sampler
from gfs.models.samplers.sfess.sfess import score_function_estimator

from functools import partial

class MLP(nn.Module):
    def __init__(self, in_dim=100, out_dim=10):
        super().__init__()

        self.num_samples = 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),  
        )

    def forward(self, x, epoch, total_epochs, training):
        logits = self.mlp(x).squeeze(-1)
        return logits

# -------------------------
# 1) Data: 5 features, first 2 informative
# -------------------------
X, y = make_classification(
    n_samples=20000,
    n_features=500,
    n_informative=20,
    n_redundant=0,
    n_repeated=0,
    n_classes=10,
    n_clusters_per_class=1,
    shuffle=False,     # informative are columns 0 and 1
    class_sep=0.25, # default is 1.0 — smaller = weaker signal
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=512, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(in_dim=X.shape[1], out_dim=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

def acc_from_logits(logits, y_true):
    preds = torch.argmax(logits, dim=1)
    return (preds == y_true).float().mean().item()

# -------------------------
# 2) Train
# -------------------------
total_epochs = 200
for epoch in range(1, total_epochs):
    model.train()
    tr_loss = tr_acc = 0.0
    n = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb, epoch, total_epochs, True)
        # print("crit ", logits.shape, yb.shape)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        tr_loss += loss.item() * bs
        tr_acc  += acc_from_logits(logits.detach(), yb) * bs
        n += bs

    model.eval()
    te_loss = te_acc = 0.0
    m = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, epoch, total_epochs, False)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            te_loss += loss.item() * bs
            te_acc  += acc_from_logits(logits, yb) * bs
            m += bs

    if epoch % 5 == 0 or epoch == 1:
        # "Recovered" features: top-k learned logits
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss/n:.4f} acc {tr_acc/n:.3f} | "
            f"test loss {te_loss/m:.4f} acc {te_acc/m:.3f} | "
        )

model.eval()
te_loss = te_acc = 0.0
m = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb, epoch, total_epochs, False)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        te_loss += loss.item() * bs
        te_acc  += acc_from_logits(logits, yb) * bs
        m += bs

print(
    f"Final | "
    f"test loss {te_loss/m:.4f} acc {te_acc/m:.3f} | "
)


