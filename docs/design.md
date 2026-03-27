# Design

## Goal

Learn optimal gene panels (10-50 genes) for spatial transcriptomics cell type classification. The model selects which genes to measure in post-hoc profiling experiments, where one fixed panel is applied to all cells.

## Architecture

Three composable stages, each independently configurable via Hydra:

```
Raw gene expression (n_nodes, n_genes) + spatial coords (n_nodes, 2-3)
         │
   ┌─────▼──────┐
   │  Feature    │  ← pluggable: Gumbel / STG / scGist
   │  Selection  │     outputs binary mask at eval, soft mask at train
   └─────┬──────┘
         │ masked expression (n_nodes, n_genes)
   ┌─────▼──────┐
   │    GNN     │  ← pluggable: GAT / SAGE / GCN
   │  Backbone  │     message passing over spatial graph
   └─────┬──────┘
         │ node embeddings (n_nodes, hid_ch)
   ┌─────▼──────┐
   │   Task     │  ← pluggable: classification / reconstruction
   │   Head(s)  │
   └─────┬──────┘
         │ predictions
```

## Key constraints

### 1. Hard masks at val/test

At evaluation, feature masks must be binary (0/1). Continuous masks leak information by allowing partial signal from "unselected" genes.

| Method | Train | Val/Test |
|--------|-------|----------|
| Gumbel | Soft Gumbel-softmax | Hard argmax → binary |
| STG | Soft hard_sigmoid(mu + noise) | Top-k by mu → binary |
| scGist | Continuous logits | Top-k by logit → binary |

### 2. Uniform mask within subgraphs

All nodes in a subgraph share the same mask. Different subgraphs in a batch may receive different sampled masks (stochastic training exploration). This matches deployment: one panel for all cells.

- Gumbel: already per-subgraph. Correct.
- STG: currently per-batch. Add per-subgraph noise sampling for more gradient diversity.
- scGist: deterministic. No change needed.

### 3. Seed-node-only evaluation

Each batch entry is a local patch centered on a seed node. At val/test, only the seed node's prediction counts for metrics. Peripheral nodes provide GNN message-passing context only.

Training is configurable via `trainmode`:
- `trainmode=0`: backprop through all train-masked nodes in the batch
- `trainmode=1`: backprop through seed nodes only

### 4. Inductive train/test split

Train on one hemisphere, test on the other. The two hemispheres of a coronal section share cell types but are spatially independent graphs. This matches deployment: selecting genes before profiling new tissue.

Splitting is done in preprocessing (not in training code) since procedures vary per brain/dataset.

### 5. Modular task heads

Support multiple downstream objectives:
- **Classification**: node embeddings → class logits (CE / focal loss)
- **Reconstruction**: node embeddings → predicted full gene expression (MSE loss)

Multi-task training (joint classification + reconstruction) is a future goal.

## Hydra config structure

```
conf/
├── config.yaml              # top-level defaults + global flags
├── data/
│   └── hemisphere.yaml
├── backbone/                 # GNN architecture
│   ├── gat.yaml
│   ├── sage.yaml
│   └── gcn.yaml
├── feature_selection/        # feature selection method
│   ├── gumbel.yaml
│   ├── stg.yaml
│   └── scgist.yaml
├── task/                     # downstream task head
│   ├── classification.yaml
│   └── reconstruction.yaml
├── trainer/
│   └── default.yaml
└── logging/
    └── default.yaml
```

Global flags (in `config.yaml`):
- `n_select`: number of genes to select
- `per_subgraph_mask`: whether each subgraph gets independent mask samples
- `trainmode`: 0 (all nodes) or 1 (seed only)
- `halfhop`: HalfHop augmentation toggle

Usage:
```bash
python -m gfs.trainers.train backbone=gat feature_selection=stg task=classification
python -m gfs.trainers.train backbone=sage feature_selection.sigma=0.3 n_select=20
```

## Testing philosophy

Integration tests, not unit tests. Each test loads real data from `data/dev/` and exercises the full path through the component being built. Tests are written alongside (or before) the code they validate — not as an afterthought.

Examples:
- Data pipeline test: load h5ad → build PyG Data → check shapes, dtypes, label encoding, no cross-split leakage
- Feature selection test: create real-shaped input from dev data → forward pass → verify mask is binary at eval, continuous at train, same within subgraph
- Backbone test: load dev graph → feature selection → GNN forward → verify embedding shapes
- End-to-end test: full training step on dev data → loss decreases

Tests live in `tests/` and run with `pytest`. They share a common fixture that loads the dev dataset once.

Additionally, `tests/featselect/` contains functional tests that verify feature selectors on synthetic data (via `make_classification`). These test recovery of informative features, comparison against random baselines, and temperature/noise parameter behavior. Marked `@pytest.mark.slow` since they train small models.

## Approach

Clean rewrite from these abstractions. Archive `src/gfs/` as `src/gfs_archive/` for reference. Build incrementally with integration tests at each step, using real data from `data/dev/`.
