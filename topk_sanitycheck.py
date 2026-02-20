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

class SubsetLayer(nn.Module):

    def __init__(self, subset_layer, k, num_samples):
        super(SubsetLayer, self).__init__()
        self.subset = subset_layer
        self.k = k
        self.num_samples = num_samples

    def forward(self, logits, tau, training):
        if training:
            res = self.subset(logits, tau)
            return res
        else:
            indices = torch.topk(logits.squeeze(-1), self.k, dim=1)[1]
            khot = F.one_hot(indices, logits.size(1)).sum(1).float()
            khot = khot.unsqueeze(0).unsqueeze(-1).expand(self.num_samples, -1, -1, -1)
            return khot, None


def get_subset_layer(k, args):
    name = {
        "sfess": "sfess",
        "sfess-v": "sfess",
        "gumbel": "gumbel",
        "st-gumbel": "gumbel",
        "simple": "simple",
        "imle": "imle",
        "pps": "pps",
    }[args.sampler]
    print("SAMPLER: ", name)
    sampler_args = SamplerArgs(
        name=name,
        sample_k=k,
        n_samples=args.num_samples,
        noise_scale=args.noise_scale,
        beta=args.beta,
        tau=args.tau,
        hard=args.sampler != "gumbel",
        pps_gradient=args.pps_gradient,
        pps_activation=args.pps_activation,
        pps_sample=args.pps_sample,
    )
    sampler = get_sampler(sampler_args, device=args.device)
    subset_layer = SubsetLayer(sampler, k, args.num_samples)
    return subset_layer

def get_sfe(args):
    estimator = {
        "sfess": "reinforce",
        "sfess-v": "vimco",
        "gumbel": None,
        "st-gumbel": None,
        "simple": None,
        "imle": None,
        "pps": None,
    }[args.sampler]
    return partial(score_function_estimator, estimator=estimator)




class TestArgs:
    def __init__(self):
        self.sampler = "pps"
        self.num_samples = 1
        self.noise_scale = 1.0
        self.beta = 1.0
        self.tau = 0.1
        self.pps_sample = "pareto"
        self.pps_activation = "scaled_sigmoid"
        self.pps_gradient = "straight_through"
        # self.device = "cuda"
        self.device = "cpu"



# -------------------------
# Model: Gate + MLP
# -------------------------
class GatedMLP(nn.Module):
    def __init__(self, in_dim=100, out_dim=10, k=10):
        super().__init__()

        self.cfg_topk = TestArgs()
        self.k = k
        self.num_samples = self.cfg_topk.num_samples
        self.subset_layer = get_subset_layer(k, self.cfg_topk)
        self.logits = nn.Parameter(torch.randn(in_dim))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),  # logits
        )

    def sample_mask(self, logits, batch, tau, training):
        if logits.dim() == 1:
            logits = logits.expand(batch, -1)


        if training:
            mask, extra = self.subset_layer(logits.unsqueeze(-1), tau, training)
            # print("MASK295295 ", mask.shape) MASK295295  torch.Size([1, 1, 500])
            # print("MASK295HARD ", mask) 10 1s,490 0s when hard sampling is used
            self.extra = extra #??
            # print("EXTRA? ", self.extra)
            mask = mask.squeeze()
            # print("trainmask ", mask.shape)
            return mask
        else:
            indices = torch.topk(logits, self.k, dim=-1)[1] 
            mask = F.one_hot(indices, logits.size(-1)).sum(1)
            mask = mask.float()
            mask = mask.expand(self.num_samples, -1, -1)
            mask = mask.squeeze()
            print("testmask ", mask)
            return mask


    def forward(self, x, epoch, total_epochs, training):
        mask = self.sample_mask(self.logits, 1, tau_schedule("exp", epoch , total_epochs), training)
        # print("beforemask ", x.shape)
        x = mask * x
        # print("aftermask ", x.shape)
        logits = self.mlp(x).squeeze(-1)
        return logits, mask

def tau_schedule(type, epoch, total_epoch):
    start_tau = 10
    end_tau = 0.01 
    # end_tau = 0.1

    if type == 'exp':
        tau = start_tau * (end_tau / start_tau) ** (epoch / total_epoch)
    elif type == "constant":
        tau = 0.1
    return tau
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
model = GatedMLP(in_dim=X.shape[1], out_dim=10, k=10).to(device)

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
        logits, mask = model(xb, epoch, total_epochs, True)
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
            logits, mask = model(xb, epoch, total_epochs, False)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            te_loss += loss.item() * bs
            te_acc  += acc_from_logits(logits, yb) * bs
            m += bs

    if epoch % 5 == 0 or epoch == 1:
        # "Recovered" features: top-k learned logits
        learned = torch.topk(model.logits.detach().cpu(), k=10).indices.numpy()
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss/n:.4f} acc {tr_acc/n:.3f} | "
            f"test loss {te_loss/m:.4f} acc {te_acc/m:.3f} | "
            f"top10 logits idx {learned}"
        )

model.eval()
te_loss = te_acc = 0.0
m = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, mask = model(xb, epoch, total_epochs, False)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        te_loss += loss.item() * bs
        te_acc  += acc_from_logits(logits, yb) * bs
        m += bs

print(
    f"Final | "
    f"test loss {te_loss/m:.4f} acc {te_acc/m:.3f} | "
)

print("\nFinal learned feature logits:", model.logits.detach().cpu().numpy())
probs = F.softmax(model.logits.detach().cpu())
print("\nFinal learned feature probs:", probs)
values, indices = torch.topk(probs, k=10)
print("Learned feature index: ", indices)  

print("Expected informative features are [0...9] (because shuffle=False).")
