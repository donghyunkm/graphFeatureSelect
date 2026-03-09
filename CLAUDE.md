# GFSNet - Graph-based Feature Selection

---

## Project overview: only the user modifies this section.

GFSNet learns optimal gene panels for spatial transcriptomics using graph neural networks with differentiable feature selection. The system helps neuroscientists select which genes to measure (typically 10-50) in post-hoc profiling experiments.

The broad goal is to go over different variants of feature selection layers and graph neural network implementations for the classification task.

The main entry point is:
`src/gfs/trainers/train.py`

Feature selection method is selected via Hydra config:
- `model=antelope` (Gumbel softmax / scGist)
- `model=antelope_stg` (Stochastic Gates)

---

## Project Structure

```
gfsnet/
├── src/
│   └── gfs/                        # Main package
│       ├── __init__.py
│       ├── conf/                    # Hydra config (composable groups)
│       │   ├── config.yaml          # top-level defaults list
│       │   ├── data/
│       │   │   └── hemisphere.yaml
│       │   ├── model/
│       │   │   ├── antelope.yaml
│       │   │   └── antelope_stg.yaml
│       │   ├── trainer/
│       │   │   └── default.yaml
│       │   └── logging/
│       │       └── default.yaml
│       ├── data/
│       │   ├── __init__.py
│       │   └── hemisphere.py        # PyGAnnData, DataModule
│       ├── models/
│       │   ├── __init__.py
│       │   ├── backbone.py          # GnnFs (unified GNN + feature selection)
│       │   ├── lit_module.py        # LitGnnFs Lightning module (unified)
│       │   ├── components.py        # MLP, FeatureRegularizer
│       │   ├── feature_selection/
│       │   │   ├── __init__.py      # registry / factory
│       │   │   ├── gumbel.py        # persist/scGist (Gumbel softmax mask)
│       │   │   └── stg.py           # STG FeatureSelector
│       │   ├── samplers/            # Differentiable sampling schemes
│       │   │   └── ...
│       │   ├── stg/                 # Original STG layers/utils (kept for reference)
│       │   │   ├── layers.py
│       │   │   └── utils.py
│       │   └── transforms.py        # HalfHop
│       ├── trainers/
│       │   ├── __init__.py
│       │   └── train.py             # Single unified entry point
│       └── utils.py
├── scripts/                         # SLURM job scripts
│   ├── main.sh
│   └── mainconfig.sh
├── docs/                            # Documentation
│   ├── dev.md
│   ├── model.md
│   ├── experiments.md
│   └── data-description.md
├── tests/                           # Test suite
│   ├── test_imports.py
│   ├── test_config.py
│   ├── test_model_assembly.py
│   └── test_lit_module.py
├── pyproject.toml                   # Package config (Hatchling + uv, src layout)
├── config.toml                      # Runtime paths
├── CLAUDE.md
└── README.md
```

## Documentation

- **[Development Guide](docs/dev.md)** - Environment setup, build system, code style
- **[Model Architecture](docs/model.md)** - Model variants, training details, feature selection methods
- **[Running Experiments](docs/experiments.md)** - Configuration, cross-validation, output analysis
- **[Data Description](docs/data-description.md)** - Dataset information

## Core Components

### Unified Model

Single GNN model (`models/backbone.py: GnnFs`) with composable feature selection:
- **persist** - Gumbel softmax k-hot mask (`feature_selection/gumbel.py`)
- **scGist** - Continuous logits with regularizer (`feature_selection/gumbel.py`)
- **stg** - Stochastic Gates (`feature_selection/stg.py`)

Feature selection method is chosen via config (`model.fs_method`) and instantiated by `feature_selection/__init__.py: get_feature_selector()`.

### Data Pipeline

- Converts AnnData (h5ad) to PyTorch Geometric format
- Spatial graph construction with k-hop neighborhoods
- Stratified k-fold cross-validation (5 folds)
- Mini-batch sampling with `NeighborLoader`

### Configuration

Uses Hydra with composable config groups:

```yaml
# Override model variant:
python -m gfs.trainers.train model=antelope
python -m gfs.trainers.train model=antelope_stg

# Override specific params:
python -m gfs.trainers.train model.n_select=20 trainer.max_epochs=100
```

See [docs/experiments.md](docs/experiments.md) for complete configuration options.

## Environment

- Conda env: `gfsnet` (Python 3.12)
- Package manager: **uv** (not pip)
- Install: `conda activate gfsnet && uv pip install -e ".[dev,mldep,data]"`
- Run: `conda run -n gfsnet python ...`

## Tech Stack

- **PyTorch**, **PyTorch Geometric**, **Lightning** - Deep learning
- **Hydra** - Configuration management
- **AnnData** - Spatial transcriptomics data
- **Hatchling + uv** - Modern Python packaging (src layout)
