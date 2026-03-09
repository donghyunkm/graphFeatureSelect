# Model Architecture

## Overview

GFSNet uses graph neural networks with differentiable feature selection to learn optimal gene panels for spatial transcriptomics cell type classification.

## Base Model: GnnFs

Located in `gfs/models/antelope.py`

### Architecture Components

**Feature Selection Layer**
- Learns which genes to select for the classification task
- Two methods available:
  - **Persist**: k-hot selection with learnable logits
  - **scGist**: Continuous gating with regularization

**GNN Backbone**
- Supports multiple GNN architectures:
  - **GAT** (Graph Attention Network) - Default
  - **GCN** (Graph Convolutional Network)
  - **SAGE** (GraphSAGE)
- Configurable depth (default: 2 layers)
- Residual connections, layer normalization, skip connections (JK)

**Spatial Integration**
- MLP head for spatial coordinates (x, y, z)
- Added to final predictions for location-aware classification

### Feature Selection Methods

#### Persist (Default)

**Method**: Learns k separate categorical distributions over genes
- Uses Gumbel-Softmax for differentiable sampling
- Temperature annealing: 10.0 → 0.01 during training
- Samples one gene per selection slot (n_select slots)
- Combines selections using max operation

**Training**: Soft k-hot masks (Gumbel-Softmax)
**Evaluation**: Hard k-hot masks (argmax)

**Advantages**:
- Guarantees exactly k genes selected
- Clean discrete selections at test time
- Works well with small panel sizes (10-20 genes)

#### scGist

**Method**: Continuous gating with panel size regularization
- Learns a continuous weight per gene
- Regularization enforces sparsity and panel size constraint
- Regularization components:
  - Force weights toward 0 or 1
  - Panel size constraint (soft or strict)
  - Pairwise selection penalties (optional)

**Advantages**:
- Smoother optimization landscape
- Can incorporate prior knowledge via pairwise constraints
- More flexible for varying panel sizes

## Model Variants

### 1. Antelope STG (`antelope_stg.py`)

**Feature Selection**: Stochastic Gates
- Uses STG layers for feature selection
- Probabilistic approach with reparameterization trick
- Can learn sparse selections without hard constraints

## Training Details

### Forward Pass

1. **Feature Selection**: Generate soft k-hot mask
   - Training: `get_mask(tau, subgraph_id)` with Gumbel-Softmax
   - Evaluation: Hard k-hot mask via argmax
2. **Masking**: Apply mask to gene expression features
3. **Spatial Processing**: Optional MLP for spatial coordinates
4. **GNN Layers**: Message passing on graph structure
5. **Prediction**: Final linear layer for classification
6. **Residual**: Add spatial and gene expression residuals

### Loss Functions

- **Cross-Entropy** (default): Standard classification loss
- **Focal Loss** (optional): For imbalanced datasets
  - `alpha=0.25`, `gamma=2.0`
- **Regularization** (scGist only): Panel size and sparsity constraints

### Classification metrics to track

**Training**:
- Overall accuracy (weighted)
- Macro accuracy

**Validation**:
- Overall accuracy (weighted)
- Macro accuracy
- Cross-entropy loss

**Test**:
- Overall accuracy (weighted)
- Macro accuracy
- Micro accuracy
- F1 scores (overall, macro, micro)

## Data Pipeline

### PyGAnnData (`gfs/data/hemisphere.py`)

Converts AnnData objects to PyTorch Geometric format:

**Input**: AnnData with:
- `.X`: Gene expression matrix (n_cells × n_genes)
- `.obs[cell_type]`: Cell type labels
- `.obs[spatial_coords]`: Spatial coordinates (x, y, z)
- `.obsp['spatial_connectivities']`: Spatial graph connectivity

**Output**: PyG Data with:
- `x`: Concatenated gene expression + spatial coords
- `edge_index`: Graph connectivity
- `celltype`: Encoded cell type labels
- `train_mask`, `val_mask`, `test_mask`: Split indicators
- `gene_exp_ind`, `xyz_ind`: Feature indices

### Data Augmentation

**HalfHop Transform** (`gfs/models/transforms.py`):
- Randomly removes "slow" edges during training
- Forces model to learn robust representations
- Optional: controlled by `halfhop` config parameter

### Batching

**NeighborLoader**:
- Samples k-hop neighborhoods around root nodes
- Creates mini-batches of subgraphs
- Tracks `subgraph_id` for independent mask generation
- Default: 2-hop neighborhoods, batch_size=64

## Configuration

Key model parameters in `gfs/configs/antelope.yaml`:

```yaml
model:
  name: "antelope"
  gene_ch: 500              # Number of input genes
  spatial_ch: 3             # Spatial dimensions (x, y, z)
  hid_ch: 32                # Hidden dimension
  out_ch: 158               # Number of cell types
  n_select: 10              # Number of genes to select
  local_layers: 2           # GNN depth
  dropout: 0.5              # Dropout rate
  heads: 1                  # Attention heads (GAT only)
  gnn: "gat"                # "gat", "sage", or "gcn"
  fs_method: "persist"      # "persist" or "scGist"
  tautype: "exp"            # Temperature schedule: "exp" or "constant"
  focal_loss: false         # Use focal loss for imbalanced data
```

## Important Implementation Notes

- Seed nodes are the root nodes in each batch; neighbors are included for message passing
- The `subgraph_id` field tracks which seed node each neighbor belongs to
- Separate masks are generated for each subgraph during training
- Gene selections are logged per epoch to `selections.csv` for monitoring convergence
- Temperature annealing is critical for discrete feature selection to work properly
