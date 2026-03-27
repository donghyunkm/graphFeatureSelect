# Model Architecture

## Overview

GFSNet uses graph neural networks with differentiable feature selection to learn optimal gene panels for spatial transcriptomics cell type classification. The architecture is composed of three independently composable stages:

1. **Feature Selection** -- selects a subset of genes via a learnable mask
2. **GNN Backbone** -- produces node embeddings from masked expression + spatial coordinates
3. **Task Head** -- maps embeddings to predictions (classification or reconstruction)

Each stage is a separate `nn.Module` wired together by the Lightning module (`LitGnnFs`). Stages are configured independently via Hydra config groups (`feature_selection/`, `backbone/`, `task/`), so any feature selector works with any backbone and any task head.

## Architecture Diagram

```
Input: gene_exp (n_nodes, n_genes)    xyz (n_nodes, spatial_ch)    subgraph_id (n_nodes,)
              |                              |                              |
     +--------v-----------+                  |                              |
     | Feature Selector   |                  |                              |
     | (gumbel/stg/scgist)|                  |                              |
     +---------+----------+                  |                              |
               |  masked_exp                 |                              |
               |  (n_nodes, n_genes)         |                              |
     +---------v-----------------------------v------------------------------v--------+
     |                          GNN Backbone                                         |
     |                                                                               |
     |   [pre_linear]     gene_ch -> hid_ch                                          |
     |        |                                                                      |
     |   [GNN layers]     x N with residual + LayerNorm + dropout + JK               |
     |        |                                                                      |
     |   [+ x_res_mlp]    MLP(masked_exp) -> hid_ch         (optional)               |
     |   [+ xyz_proj]     Linear(centered_xyz) -> hid_ch     (optional)              |
     |                                                                               |
     +--------------------------------------+----------------------------------------+
                                            |  embeddings (n_nodes, hid_ch)
                                    +-------v--------+
                                    |   Task Head    |
                                    | (cls or recon) |
                                    +-------+--------+
                                            |
                                     output (logits or predicted expression)
```

## Feature Selection Methods

All feature selectors inherit from `FeatureSelector` (abstract base class) and share a common interface:

```python
class FeatureSelector(ABC, nn.Module):
    def forward(self, x, tau, subgraph_id) -> torch.Tensor:  # masked expression
    def get_mask(self, tau, subgraph_id) -> torch.Tensor:     # current mask
    def regularization_loss(self) -> torch.Tensor:            # reg term (default 0)
    def selected_indices(self) -> torch.Tensor:               # eval-time gene indices
```

**Key design constraint**: all selectors produce **hard binary masks at eval time** (no information leakage through continuous weights). At train time, masks are soft/stochastic for gradient flow.

| Method | Config | Train mask | Eval mask | Regularization |
|--------|--------|-----------|-----------|----------------|
| Gumbel | `feature_selection=gumbel` | Gumbel-softmax k-hot per subgraph | argmax binary k-hot | None (0) |
| STG | `feature_selection=stg` | hard_sigmoid(mu + noise) per subgraph | top-k by mu, binary | Gaussian CDF sparsity |
| scGist | `feature_selection=scgist` | continuous logits (broadcast) | top-k by abs(logits), binary | Binary pressure + panel size |

**Source**: `src/gfs/models/feature_selection/`

### Gumbel Softmax (`GumbelFeatureSelector`)

**File**: `src/gfs/models/feature_selection/gumbel.py`

Learns `n_select` categorical distributions over genes via a `(n_select, n_genes)` logits parameter. Each row ("slot") selects one gene.

**Training** -- soft k-hot via Gumbel-Softmax:
1. Logits are expanded per subgraph: `(n_select, n_subgraphs, n_genes)`
2. `F.gumbel_softmax(logits, tau=tau, hard=False)` produces a soft one-hot per slot per subgraph
3. `max(dim=0)` across slots yields a soft k-hot mask per subgraph: `(n_subgraphs, n_genes)`
4. Mask is broadcast to nodes via `subgraph_id`: `k_hot[subgraph_id]`

**Evaluation** -- hard binary:
- `argmax` per slot produces `n_select` gene indices
- Scatter to binary `(1, n_genes)` mask, broadcast to all nodes

**Temperature schedule** (`tautype`):
- `"exp"`: `10.0 * (0.01 / 10.0)^(epoch / max_epochs)` -- anneals from 10.0 to 0.01
- `"constant"`: fixed at 0.1

**Config** (`feature_selection/gumbel.yaml`):
```yaml
method: "gumbel"
tautype: "exp"
```

### STG -- Stochastic Gates (`STGFeatureSelector`)

**File**: `src/gfs/models/feature_selection/stg.py`

Learns a `mu` parameter per gene (`n_genes` values). During training, adds independent Gaussian noise per subgraph.

**Training**:
```
noise ~ N(0, 1)  per subgraph       # (n_subgraphs, n_genes)
z = mu + sigma * noise
gate = clamp(z + 0.5, 0, 1)         # hard sigmoid
masked_x = x * gate[subgraph_id]    # per-node mask via subgraph_id
```

**Evaluation** -- hard binary:
- Top-k genes by `mu` value, scatter to binary `(1, n_genes)` mask

**Regularization**: Gaussian CDF evaluated at `(mu + 0.5) / sigma`, averaged over genes. Penalizes open gates -- minimizing pushes `mu` below the threshold so gates close. Weighted by `lam` in total loss.

**Config** (`feature_selection/stg.yaml`):
```yaml
method: "stg"
sigma: 0.5       # noise scale; lower = sharper gates
tautype: "constant"
```

### scGist -- Continuous Gating (`ScGistFeatureSelector`)

**File**: `src/gfs/models/feature_selection/gumbel.py`

Learns a `(1, n_genes)` logits parameter initialized to 0.5. Simplest selector -- expression is multiplied by continuous weights at train time.

**Training**: `masked_x = x * logits` (continuous, broadcasts over nodes)

**Evaluation** -- hard binary:
- Top-k genes by `abs(logits)`, scatter to binary `(1, n_genes)` mask

**Regularization** (two terms, weighted by `l1`):
- **Binary pressure**: `sum(|w| * |w - 1|)` -- pushes weights toward 0 or 1
- **Panel size**: `|sum(|w|) - n_select|` -- constrains total active features

**Config** (`feature_selection/scgist.yaml`):
```yaml
method: "scgist"
l1: 0.1          # regularization weight for binary + size penalties
tautype: "constant"
```

## GNN Backbone (`GNNBackbone`)

**File**: `src/gfs/models/backbone.py`

A standalone module that takes masked gene expression and spatial coordinates, runs them through a configurable GNN stack, and returns node embeddings of dimension `hid_ch`.

### Layer Factory

The `_build_gnn_layer()` factory supports three GNN architectures:

| Type | PyG Class | Notes |
|------|-----------|-------|
| `"gat"` | `GATv2Conv` | Attention-based; configurable heads, no self-loops |
| `"sage"` | `SAGEConv` | Sampling-friendly aggregation |
| `"gcn"` | `GCNConv` | Spectral convolution, normalized |

### Per-Layer Options

Each GNN layer can optionally include:
- **Residual connection**: `GNN(x) + Linear(x)` (skip connection with learned projection)
- **LayerNorm** or **BatchNorm**: normalization after each layer
- **JK (Jumping Knowledge)**: sum outputs from all layers instead of using only the last
- **Dropout**: applied after activation

### Spatial Coordinate Centering

When `subgraph_id` is provided, XYZ coordinates are centered per subgraph using scatter-based mean subtraction:

```python
means = scatter_mean(xyz, subgraph_id, dim=0)
xyz_centered = xyz - means[subgraph_id]
```

This makes spatial features translation-invariant within each sampled patch.

### Residual Paths

Two optional paths add information to the final GNN output:

- **xyz_proj**: `Linear(spatial_ch -> hid_ch)` -- projects centered spatial coordinates
- **x_residual**: `MLP(gene_ch -> 128 -> 128 -> hid_ch)` -- residual from masked expression, bypassing the GNN stack

### Config (`backbone/gat.yaml` example):

```yaml
gnn_type: "gat"
hid_ch: 32
n_layers: 2
dropout: 0.5
heads: 1
pre_linear: true
residual: true
layer_norm: true
batch_norm: false
jk: true
xyz_proj: true
x_residual: true
```

Other backbone configs (`backbone/sage.yaml`, `backbone/gcn.yaml`) swap `gnn_type` and adjust defaults as needed.

## Task Heads

**File**: `src/gfs/models/heads.py`

### Classification Head (`ClassificationHead`)

Single linear layer: `Linear(hid_ch -> n_classes)`. Returns logits.

Config (`task/classification.yaml`):
```yaml
name: "classification"
loss: "ce"
focal_loss: false
```

Loss: `nn.CrossEntropyLoss()` (or focal loss when `focal_loss: true`).

### Reconstruction Head (`ReconstructionHead`)

MLP: `Linear -> ReLU -> Linear -> ReLU -> Linear`, mapping `hid_ch` to `n_genes`. Predicts the full (unmasked) expression profile from embeddings. Used to validate that selected genes carry enough information.

Config (`task/reconstruction.yaml`):
```yaml
name: "reconstruction"
hidden: [128, 128]
```

Loss: `nn.MSELoss()`.

## Lightning Module (`LitGnnFs`)

**File**: `src/gfs/models/lit_module.py`

Wires the three components together and handles training, validation, and test loops.

### Two-Phase Initialization

1. **`__init__(config)`** -- saves hyperparameters but does NOT instantiate model components. At this point `n_genes` and `n_classes` are unknown (they come from data after filtering and label encoding).

2. **`setup_model(n_genes, n_classes)`** -- called by the training script after `DataModule.setup()` provides the actual dimensions. Instantiates:
   - `self.feature_selector` via `get_feature_selector()` factory
   - `self.backbone` as `GNNBackbone`
   - `self.task_head` as `ClassificationHead` or `ReconstructionHead`
   - Metrics (torchmetrics `MulticlassAccuracy`, `MulticlassF1Score`)

### Forward Pass

```python
def forward(gene_exp, edge_index, xyz, subgraph_id=None, tau=1.0):
    masked_exp = self.feature_selector(gene_exp, tau, subgraph_id)
    embeddings = self.backbone(masked_exp, edge_index, xyz, subgraph_id)
    return self.task_head(embeddings)
```

### Loss Computation

```python
total_loss = task_loss + lam * feature_selector.regularization_loss()
```

- For Gumbel: `regularization_loss()` returns 0, so total loss is just the task loss
- For STG: adds `lam * mean(gaussian_cdf(...))`
- For scGist: adds `lam * l1 * (binary_reg + size_reg)`

### Tau Scheduling

The `tau_schedule()` function computes the Gumbel-softmax temperature at each epoch:

- `"exp"`: `start * (end / start) ^ (epoch / max_epochs)` with start=10.0, end=0.01
- `"constant"`: fixed at 0.1

Only matters for Gumbel feature selection; STG and scGist ignore tau.

### Seed-Node-Only Evaluation

At validation and test time, loss and metrics are computed only on seed (root) nodes. `NeighborLoader` places seed nodes at indices `0..batch_size-1` in each batch, accessed via `batch.input_id.size(0)`.

At train time, `trainmode` controls behavior:
- `trainmode=0`: loss on all nodes with `train_mask`
- `trainmode=1`: loss on seed nodes only

### DRY Logging

All metrics flow through `_log_metrics()`, which reads logging options from the `logging` config group:

```yaml
# logging/default.yaml
on_step: false
on_epoch: true
prog_bar: true
logger: true
```

### Metrics

| Split | Metrics |
|-------|---------|
| Train | weighted accuracy, macro accuracy, loss, reg_loss, tau |
| Val | weighted accuracy, macro accuracy, loss |
| Test | weighted accuracy, macro accuracy, macro F1, loss |

### Optimizer

Adam with configurable learning rate. Optional `MultiStepLR` scheduler (milestones at epoch 100, gamma 0.1) when `trainer.lr_scheduler: "multistep"`.

## Configuration

Hydra composable config groups under `src/gfs/conf/`:

```
src/gfs/conf/
  config.yaml               # defaults + global flags
  backbone/
    gat.yaml                 # GATv2Conv defaults
    sage.yaml                # SAGEConv defaults
    gcn.yaml                 # GCNConv defaults
  feature_selection/
    gumbel.yaml              # Gumbel-softmax
    stg.yaml                 # Stochastic Gates
    scgist.yaml              # scGist continuous
  task/
    classification.yaml      # Linear head + CE loss
    reconstruction.yaml      # MLP head + MSE loss
  data/
    hemisphere.yaml          # Inductive hemisphere split
  trainer/
    default.yaml             # Epochs, LR, batch limits
  logging/
    default.yaml             # Log timing and destinations
```

**Global flags** in `config.yaml`:
```yaml
n_select: 10       # number of genes to select
trainmode: 0       # 0=all nodes, 1=seed only
halfhop: false     # HalfHop graph augmentation
lam: 0.1           # feature selection regularization weight
```

Compose any combination on the command line:
```bash
python -m gfs.trainers.train \
  backbone=gat feature_selection=stg task=classification \
  n_select=20 lam=0.05
```

## Important Implementation Notes

- **`subgraph_id`**: Every node in a `NeighborLoader` batch is assigned to its seed node's subgraph. This is critical for Gumbel and STG to generate one mask per subgraph (uniform within patches).
- **Seed nodes first**: `NeighborLoader` places seed nodes at batch indices `0..input_id.size(0)-1`. The Lightning module uses this convention for seed-node-only eval.
- **Hard binary masks at eval**: All three feature selectors produce deterministic binary masks at eval time. This is a design constraint to prevent information leakage through continuous mask values.
- **Inductive split**: Train data uses one hemisphere, test uses the other. The split is done at the data level (separate h5ad files), not via masks in the training code.
- **Scatter-based XYZ centering**: Spatial coordinates are mean-centered per subgraph, making the model invariant to absolute position within the section.
- **Two-phase init**: `LitGnnFs.__init__()` only saves config; `setup_model()` builds the actual modules after data dimensions are known. This avoids hardcoding gene/class counts in config.
