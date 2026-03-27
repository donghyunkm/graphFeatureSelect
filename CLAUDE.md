# GFSNet - Graph-based Feature Selection

---

## Project overview: only the user modifies this section.

GFSNet learns optimal gene panels for spatial transcriptomics using graph neural networks with differentiable feature selection. The system helps neuroscientists select which genes to measure (typically 10-50) in post-hoc profiling experiments.

The broad goal is to go over different variants of feature selection layers and graph neural network implementations for the classification task.

The main entry point is:
`src/gfs/trainers/train.py`

Feature selection method is selected via Hydra config:
- `feature_selection=gumbel` (Gumbel softmax)
- `feature_selection=stg` (Stochastic Gates)
- `feature_selection=scgist` (scGist continuous gating)

---

## Architecture

Three composable stages, independently configurable via Hydra config groups:

```
Gene expression (n_nodes, n_genes) + spatial coords (n_nodes, 2)
         │
   ┌─────▼──────┐
   │  Feature    │  ← backbone/: gat, sage, gcn
   │  Selection  │     Hard binary masks at eval
   └─────┬──────┘
         │ masked expression (n_nodes, n_genes)
   ┌─────▼──────┐
   │    GNN     │  ← feature_selection/: gumbel, stg, scgist
   │  Backbone  │     Scatter-based XYZ centering
   └─────┬──────┘
         │ node embeddings (n_nodes, hid_ch)
   ┌─────▼──────┐
   │   Task     │  ← task/: classification, reconstruction
   │   Head     │
   └─────┬──────┘
         │ predictions
```

### Key design constraints
1. **Hard binary masks at val/test** — all selectors produce 0/1 masks at eval
2. **Uniform mask within subgraphs** — all nodes in a patch share one mask
3. **Seed-node-only eval** — only central node counts for val/test metrics
4. **Inductive hemisphere split** — train on one hemisphere, test on the other
5. **Modular task heads** — classification and reconstruction, composable via Hydra

## Project Structure

```
gfsnet/
├── src/
│   └── gfs/
│       ├── conf/                        # Hydra config
│       │   ├── config.yaml              # Top-level defaults + global flags
│       │   ├── data/hemisphere.yaml     # Data paths and loading config
│       │   ├── backbone/                # GNN architecture (gat, sage, gcn)
│       │   ├── feature_selection/       # Selection method (gumbel, stg, scgist)
│       │   └── task/                    # Task head (classification, reconstruction)
│       ├── data/
│       │   ├── dataset.py              # PyGAnnData: h5ad → PyG Data
│       │   ├── datamodule.py           # HemisphereDataModule (Lightning)
│       │   └── hemisphere.py           # Legacy data loading (reference)
│       ├── models/
│       │   ├── feature_selection/      # FeatureSelector ABC + implementations
│       │   │   ├── base.py             # Abstract base class
│       │   │   ├── gumbel.py           # Gumbel + scGist selectors
│       │   │   └── stg.py             # STG selector
│       │   ├── backbone.py            # GNNBackbone (GAT/SAGE/GCN)
│       │   ├── heads.py               # ClassificationHead, ReconstructionHead
│       │   └── lit_module.py          # LitGnnFs (Lightning module)
│       └── trainers/
│           └── train.py               # Hydra entry point
├── tests/                              # Tests (101 total)
│   ├── test_data_pipeline.py          # Data loading, shapes, splits (10)
│   ├── test_feature_selection.py      # All selectors, masks, gradients (58)
│   ├── test_backbone.py              # GNN, heads, full pipeline (16)
│   ├── test_end_to_end.py            # Training loops, all methods (6)
│   └── featselect/                    # Functional feature selection tests (11)
│       ├── conftest.py               # GatedMLP harness, toy_data fixture
│       ├── test_feature_recovery.py  # Gumbel/STG/scGist recover informative features
│       ├── test_baseline.py          # Learned mask beats random baseline
│       └── test_tau_behavior.py      # Temperature/noise controls sharpness
├── refactor/                           # Design docs
│   ├── design.md                      # Architecture and constraints
│   ├── data.md                        # Data format and preprocessing
│   ├── model.md                       # Component interfaces
│   ├── dataloader.md                  # Batch format and sampling
│   └── todo.md                        # Progress checklist
├── data/
│   ├── raw/                           # Raw h5ad files
│   └── dev/                           # Dev dataset (single section)
├── notebooks/                          # Preprocessing scripts
├── docs/                              # User-facing documentation
│   ├── model.md                       # Architecture reference
│   ├── experiments.md                 # Running experiments
│   ├── dev.md                         # Environment setup
│   └── data.md                        # Dataset description
├── pyproject.toml                     # Package config (Hatchling + uv)
└── CLAUDE.md
```

## Documentation

- **[docs/model.md](docs/model.md)** — Model architecture reference
- **[docs/experiments.md](docs/experiments.md)** — Running experiments and configuration
- **[docs/dev.md](docs/dev.md)** — Environment setup, build system, code style
- **[docs/data.md](docs/data.md)** — Dataset description
- **[refactor/](refactor/)** — Design docs (architecture, data, model, dataloader, progress)

## Environment

- Conda env: `gfsnet` (Python 3.12)
- Package manager: **uv** (not pip)
- Install: `conda activate gfsnet && uv pip install -e ".[dev,mldep,data]"`
- Run: `conda run -n gfsnet python ...`
- Tests: `conda run -n gfsnet python -m pytest` (use `-m "not slow"` to skip training-heavy tests)

## Running

```bash
# Default (GAT + Gumbel + classification)
python -m gfs.trainers.train

# Switch components
python -m gfs.trainers.train backbone=sage feature_selection=stg task=classification

# Override parameters
python -m gfs.trainers.train backbone.hid_ch=64 n_select=20 feature_selection.sigma=0.3
```

## Tech Stack

- **PyTorch**, **PyTorch Geometric**, **Lightning** — Deep learning
- **Hydra** — Configuration management
- **AnnData** — Spatial transcriptomics data
- **Hatchling + uv** — Modern Python packaging (src layout)
