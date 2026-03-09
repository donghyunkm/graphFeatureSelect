# Model Architecture

## Overview

GFSNet uses graph neural networks with differentiable feature selection to learn optimal gene panels for spatial transcriptomics cell type classification. The model is composed of three main parts: a pluggable **feature selection** layer, a **GNN backbone**, and a **prediction head** with optional spatial residual connections.

## Unified Model: GnnFs

Located in `src/gfs/models/backbone.py`

The model accepts a feature selection method via config (`model.fs_method`). The feature selector is instantiated by `get_feature_selector()` in `src/gfs/models/feature_selection/__init__.py` and injected into `GnnFs` at construction time. This means all GNN backbone code is shared — only the feature selection layer differs between variants.

### Architecture Components

```
Input: gene expression (n_nodes, n_genes) + spatial coords (n_nodes, 3)
                            │
                   ┌────────▼────────┐
                   │ Feature Selector │  ← pluggable: persist / scGist / stg
                   └────────┬────────┘
                            │ masked gene expression
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼─────┐ ┌────▼────┐ ┌──────▼──────┐
        │  x_res MLP │ │ lin_in  │ │  xyz linear │
        │ (optional) │ │(if pre) │ │  (optional) │
        └─────┬─────┘ └────┬────┘ └──────┬──────┘
              │        ┌────▼────┐        │
              │        │GNN layer│←─ edge_index
              │        │  + res  │
              │        │  + norm │
              │        │  + drop │
              │        └────┬────┘
              │             │  × local_layers (with JK skip connections)
              │        ┌────▼────┐
              │        │pred_local│
              │        └────┬────┘
              │             │
              └──── + ──────┼──── + ────┘
                            │
                      Output logits (n_nodes, n_classes)
```

**GNN Backbone** — supports three GNN architectures:
- **GAT** (`GATv2Conv`) — Graph Attention Network (default)
- **SAGE** (`SAGEConv`) — GraphSAGE
- **GCN** (`GCNConv`) — Graph Convolutional Network

Configurable features per layer:
- Residual connections (`res`): `GNN(x) + Linear(x)`
- Normalization: LayerNorm (`ln`) or BatchNorm (`bn`)
- JK skip connections (`jk`): sum outputs from all layers
- Dropout between layers

**Spatial Integration** — optional (`xyz_status: true`):
- Spatial coordinates are mean-centered per subgraph
- Linear projection added to final predictions
- `x_res`: MLP residual from raw gene expression to output

## Feature Selection Methods

Feature selectors are `nn.Module` subclasses in `src/gfs/models/feature_selection/`. Each exposes a common interface:

| Method | `forward(x, tau, subgraph_id)` | `regularization_loss()` | `on_train_epoch_end(...)` |
|--------|------|------|------|
| persist | Gumbel-softmax k-hot mask × x | 0 (no reg) | Logs selections to CSV |
| scGist | continuous logits × x | FeatureRegularizer loss × 100 | — |
| stg | hard_sigmoid(mu + noise) × x | Gaussian CDF of (mu+0.5)/sigma | Prints mask stats |

### Persist (Gumbel Softmax)

**File**: `src/gfs/models/feature_selection/gumbel.py: GumbelFeatureSelector`

**Mechanism**: Learns `n_select` categorical distributions over genes via a `(n_select, gene_ch)` logits parameter. Each row selects one gene.

**Training** — soft k-hot masks via Gumbel-Softmax:
1. Logits are replicated per subgraph in the batch: `(n_select, n_subgraphs, gene_ch)`
2. `F.gumbel_softmax(logits, tau=tau)` produces soft one-hot per selection slot
3. `torch.max(sample, dim=0)` combines slots into a single k-hot mask per subgraph
4. Mask is broadcast to nodes via `subgraph_id`

**Evaluation** — hard k-hot mask: `argmax` per selection slot, then scatter to binary mask.

**Temperature schedule** (`tautype`):
- `"exp"`: 10.0 → 0.01 over training (`start * (end/start)^(epoch/total)`)
- `"constant"`: fixed at 0.1

**Tracking**: At each epoch end, gene indices and softmax probabilities are logged to `selections.csv` for monitoring convergence.

### scGist (Continuous Gating)

**File**: `src/gfs/models/feature_selection/gumbel.py: ScGistFeatureSelector`

**Mechanism**: Learns a `(1, gene_ch)` logits parameter initialized to 0.5. Gene expression is simply multiplied by these continuous weights.

**Regularization** via `FeatureRegularizer` (in `src/gfs/models/components.py`):
- **Binary pressure**: `sum(|w| * |w - 1|)` — pushes weights toward 0 or 1
- **Panel size**: `|sum(|w|) - n_select| * alpha` — constrains total selected features
- **Pairwise penalties** (optional): discourages correlated gene pairs
- Regularization loss is scaled by `l1 * 100`

### STG (Stochastic Gates)

**File**: `src/gfs/models/feature_selection/stg.py: STGFeatureSelector`

**Mechanism**: Learns a `mu` parameter per gene. During training, adds Gaussian noise scaled by `sigma`:
```
z = mu + sigma * noise * is_training
gate = clamp(z + 0.5, 0, 1)    # hard sigmoid
x_masked = x * gate
```

**Regularization**: Gaussian CDF evaluated at `(mu + 0.5) / sigma`, averaged over genes. This measures the probability each gate is "open" — minimizing it encourages sparsity. Weighted by `lam` in the total loss.

**Key parameters**:
- `sigma`: noise scale (default 0.5 for STG, 1.0 for antelope). Lower sigma = sharper gates.
- `lam`: regularization weight (default 0.1). Higher = more aggressive pruning.

**Device handling**: The `noise` buffer is moved to the correct device via a custom `_apply()` override.

## Lightning Module: LitGnnFs

Located in `src/gfs/models/lit_module.py`

Unified Lightning module that works with all feature selection methods. Key design:

**Loss computation** (`_compute_loss`):
```python
total_loss = CE_loss + model.lam * feature_selector.regularization_loss()
```
- For persist: `regularization_loss()` returns 0, so it's just CE
- For scGist: adds `FeatureRegularizer` loss
- For STG: adds `lam * mean(gaussian_cdf(...))`

**Loss functions**:
- **Cross-Entropy** (default): `nn.CrossEntropyLoss()`
- **Focal Loss** (`focal_loss: true`): `sigmoid_focal_loss(alpha=0.25, gamma=2.0)` for imbalanced datasets

**Training modes** (`trainmode`):
- `0`: Backprop/metrics for all nodes in batch (including neighbors)
- `1`: Backprop/metrics for only root (seed) nodes

### Metrics

| Split | Metrics |
|-------|---------|
| Train | weighted accuracy, macro accuracy |
| Val | weighted accuracy, macro accuracy, CE loss |
| Test | weighted/macro/micro accuracy, weighted/macro/micro F1 |

Test predictions and labels are saved to `test_pred.pt` at the end of testing.

## Data Pipeline

### PyGAnnData (`src/gfs/data/hemisphere.py`)

Converts AnnData objects to PyTorch Geometric format:

**Input**: AnnData (`.h5ad`) with:
- `.X`: Gene expression matrix (n_cells x n_genes), dense or sparse
- `.obs[cell_type]`: Cell type labels (categorical)
- `.obs[spatial_coords]`: Spatial coordinates (e.g. x_section, y_section, z_section)
- `.obsp['spatial_connectivities']`: Precomputed spatial neighbor graph

**Preprocessing**:
- Filters blank genes (names starting with "Blank")
- Removes cell types with < 5 cells
- Encodes cell types with `LabelEncoder`
- Computes binary adjacency (self-loops removed unless `self_loops_only`)

**Train/val/test split**: `StratifiedKFold3` — a custom 3-way stratified split:
1. Standard `StratifiedKFold` produces train/test
2. Train is further split into train/val using `train_test_split`
3. Indexed by `cv` parameter (0 to n_splits-1)

**Output**: `PyGData` with:
- `x`: `[gene_exp | xyz]` concatenated (n_nodes, n_genes + 3)
- `edge_index`: Graph connectivity
- `celltype`: Encoded labels
- `gene_exp_ind`, `xyz_ind`: Index tensors to slice `x` back apart
- `train_mask`, `val_mask`, `test_mask`: Boolean split indicators

### Batching: NeighborLoaderMod

Wraps PyG's `NeighborLoader` to add `subgraph_id` — assigns each node to the seed node whose k-hop neighborhood it belongs to. This is needed for per-subgraph Gumbel-Softmax mask generation.

- Samples `[-1] * n_hops` neighbors (all neighbors at each hop)
- Default: 2-hop neighborhoods, batch_size=64

### HalfHop Transform (`src/gfs/models/transforms.py`)

Graph upsampling augmentation that slows message passing:
1. Randomly selects edges with probability `p`
2. Inserts "slow nodes" on selected edges with interpolated features: `alpha * x_src + (1-alpha) * x_dst`
3. Replaces original edges with two edges through the slow node
4. After forward pass, slow nodes are removed via `slow_node_mask`

Controlled by `model.halfhop` config parameter (default: true).

## Configuration

Hydra composable configs under `src/gfs/conf/`. Switch variants with:
```bash
python -m gfs.trainers.train model=antelope         # Gumbel softmax
python -m gfs.trainers.train model=antelope_stg      # Stochastic Gates
```

Key model parameters (`conf/model/antelope.yaml`):

```yaml
model:
  gene_ch: ${data.n_genes}    # Resolved from data config (500)
  spatial_ch: 3               # Spatial dimensions (x, y, z)
  hid_ch: 32                  # Hidden dimension
  out_ch: ${data.n_labels}    # Resolved from data config (158)
  n_select: 10                # Number of genes to select
  local_layers: 2             # GNN depth
  dropout: 0.5                # Dropout rate
  heads: 1                    # Attention heads (GAT only)
  gnn: "gat"                  # "gat", "sage", or "gcn"
  fs_method: "persist"        # "persist", "scGist", or "stg"
  tautype: "exp"              # Temperature schedule: "exp" or "constant"
  focal_loss: false           # Use focal loss for imbalanced data
  sigma: 1.0                  # STG noise scale
  lam: 0.1                    # Regularization weight
  halfhop: true               # HalfHop augmentation
  trainmode: 0                # 0=all nodes, 1=root only
  x_res: true                 # Gene expression residual connection
  xyz_status: true            # Spatial coordinate integration
```

## Important Implementation Notes

- Seed nodes are the root nodes in each batch; neighbors are included for message passing
- The `subgraph_id` field tracks which seed node each neighbor belongs to
- Gumbel-Softmax masks are generated independently per subgraph during training
- Gene selections are logged per epoch to `selections.csv` for monitoring convergence
- Temperature annealing is critical for persist feature selection to converge to discrete selections
- The `lam` parameter balances classification accuracy vs feature sparsity for STG/scGist
