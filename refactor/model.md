# Model Components

## Overview

The model has three composable stages. Each is an `nn.Module` with a defined interface, instantiated from Hydra config via factory functions.

## 1. Feature Selection (`models/feature_selection/`)

**Interface** (from `base.py`):
```python
class FeatureSelector(ABC, nn.Module):
    def forward(self, x, tau, subgraph_id) -> Tensor:
        """Apply (soft or hard) mask to input features."""

    def get_mask(self) -> Tensor:
        """Return current feature mask (binary at eval)."""

    def regularization_loss(self) -> Tensor:
        """Return regularization term for the feature selection params."""

    def on_train_epoch_end(self, logger_root_dir, current_epoch):
        """Optional logging/tracking hook."""
```

**Implementations:**

### Gumbel (`feature_selection/gumbel.py: GumbelFeatureSelector`)
- Params: `logits` of shape `(n_select, n_genes)` -- each row selects one gene
- Train: Gumbel-softmax sampling, one sample per subgraph, max across slots -> soft k-hot
- Eval: argmax per slot -> binary k-hot
- Reg: none (returns 0)
- Controlled by: `tau` (temperature), `tautype` (schedule: exp or constant)

### STG (`feature_selection/stg.py: STGFeatureSelector`)
- Params: `mu` of shape `(n_genes,)` -- gate center per gene
- Train: `hard_sigmoid(mu + sigma * noise)`, noise sampled per subgraph
- Eval: top-k genes by `mu` value -> binary mask
- Reg: `mean(gaussian_cdf((mu + 0.5) / sigma))` -- penalizes open gates
- Controlled by: `sigma` (noise scale), `lam` (reg weight)

### scGist (`feature_selection/gumbel.py: ScGistFeatureSelector`)
- Params: `logits` of shape `(1, n_genes)` -- continuous weight per gene
- Train: `x * logits` (continuous)
- Eval: top-k by logit magnitude -> binary mask
- Reg: `FeatureRegularizer` -- pushes weights to 0/1, constrains panel size
- Controlled by: `lam` (reg weight), `n_select` (panel size target)

**Factory:** `feature_selection/__init__.py: get_feature_selector(fs_method, n_genes, n_select, **kwargs)`

## 2. GNN Backbone (`models/backbone.py: GnnFs`)

Takes masked gene expression + spatial coords + edge_index, returns node embeddings.

**Architecture:**
- Optional pre-linear layer (`pre_linear`)
- Stack of GNN layers built via unified `_build_gnn_layer` factory method
- Residual connections (`res`): GNN(x) + Linear(x)
- Normalization: LayerNorm (`ln`) or BatchNorm (`bn`)
- JK skip connections (`jk`): sum all layer outputs
- Dropout between layers
- Spatial coordinate integration (`xyz_status`): linear projection added to output, normalized via scatter ops
- Expression residual (`x_res`): MLP from raw expression to output

**GNN layer types:**
- `gat`: `GATv2Conv` (attention-based)
- `sage`: `SAGEConv` (sampling + aggregation)
- `gcn`: `GCNConv` (spectral convolution)

**Note (future cleanup):** Dead code in `models/stg/` and `models/get_sampler.py` can be removed.

## 3. Task Heads (`models/heads.py`)

### ClassificationHead
- Linear: `hid_ch -> n_classes`
- Loss: CrossEntropy or focal loss
- Metrics: weighted/macro/micro accuracy and F1

### ReconstructionHead
- MLP: `hid_ch -> n_genes`
- Loss: MSE between predicted and original (unmasked) expression
- Validates that selected genes carry enough information

## Lightning Module (`models/lit_module.py: LitGnnFs`)

Orchestrates training loop:
- Wires feature selection + backbone + task head(s)
- Handles tau scheduling for Gumbel (fixed)
- Computes: `total_loss = task_loss + lam * feature_selector.regularization_loss()`
- Logs metrics per step/epoch (DRY logging helpers)
- Saves test predictions to file

## Component locations

| Component | Location | Notes |
|-----------|----------|-------|
| Feature selection | `models/feature_selection/` | Binary top-k eval masks for all methods |
| FS base class | `models/feature_selection/base.py` | `FeatureSelector` ABC |
| GNN backbone | `models/backbone.py` | Unified `_build_gnn_layer` factory |
| Task heads | `models/heads.py` | `ClassificationHead`, `ReconstructionHead` |
| Lightning module | `models/lit_module.py` | Supports multiple composable tasks |
| Components | `models/components.py` | MLP, FeatureRegularizer |
| Transforms | `models/transforms.py` | HalfHop -- keep as-is |
| Dead code | `models/stg/`, `models/get_sampler.py` | Future cleanup |
