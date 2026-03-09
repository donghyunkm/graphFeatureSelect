# GFSNet - Graph-based Feature Selection

---

## Project overview: only the user modifies this section.

GFSNet learns optimal gene panels for spatial transcriptomics using graph neural networks with differentiable feature selection. The system helps neuroscientists select which genes to measure (typically 10-50) in post-hoc profiling experiments.

The broad goal is to go over different variants of feature selection layers and graph neural network implementations for the classification task.

The main entry points for the current scope are:
`trainers/antelope.py`
`trainers/antelope_stg.py`

---

## Project Structure

```
gfsnet/
├── gfs/                    # Main package
│   ├── configs/           # Hydra experiment configs
│   │   ├── antelope.yaml
│   │   ├── antelope_top10.yaml
│   │   └── antelope_topk.yaml
│   ├── data/              # Data loading (AnnData → PyG)
│   │   └── hemisphere.py
│   ├── models/            # GNN feature selection models
│   │   ├── antelope.py
│   │   ├── antelope_stg.py
│   │   ├── samplers/      # Differentiable sampling schemes
│   │   ├── stg/           # Stochastic gates implementation
│   │   └── transforms.py
│   ├── trainers/          # Training scripts
│   │   ├── antelope.py    # Main entry point
│   │   └── antelope_stg.py # Main entry point (STG variant)
│   └── utils.py
├── scripts/               # SLURM job scripts
│   ├── main.sh            # Runs antelope.py (single job)
│   └── mainconfig.sh      # Runs antelope.py (array jobs with config)
├── docs/                  # Documentation
│   ├── dev.md            # Development setup
│   ├── model.md          # Model architecture details
│   ├── experiments.md    # Running experiments
│   └── data-description.md
├── tests/                 # Test suite
└── pyproject.toml        # Package config (Hatchling + uv)
```

## Documentation

- **[Development Guide](docs/dev.md)** - Environment setup, build system, code style
- **[Model Architecture](docs/model.md)** - Model variants, training details, feature selection methods
- **[Running Experiments](docs/experiments.md)** - Configuration, cross-validation, output analysis
- **[Data Description](docs/data-description.md)** - Dataset information

## Core Components

### Active Models

Two main feature selection approaches (see [docs/model.md](docs/model.md)):
- **Antelope** (`trainers/antelope.py`) - Default GNN with differentiable feature selection using various sampling schemes (Gumbel, IMLE, PPS, etc.)
- **Antelope STG** (`trainers/antelope_stg.py`) - GNN with Stochastic Gates (STG) for feature selection

### Data Pipeline

- Converts AnnData (h5ad) to PyTorch Geometric format
- Spatial graph construction with k-hop neighborhoods
- Stratified k-fold cross-validation (5 folds)
- Mini-batch sampling with `NeighborLoader`

### Configuration

Uses Hydra for experiment configuration:

```yaml
data:
  n_genes: 500
  n_labels: 158
  batch_size: 64
  n_hops: 2

model:
  n_select: 10        # Genes to select
  gnn: "gat"          # "gat", "sage", or "gcn"
  fs_method: "persist"

trainer:
  max_epochs: 500
  lr: 0.001
```

See [docs/experiments.md](docs/experiments.md) for complete configuration options.

## Tech Stack

- **PyTorch**, **PyTorch Geometric**, **Lightning** - Deep learning
- **Hydra** - Configuration management
- **AnnData** - Spatial transcriptomics data
- **Hatchling + uv** - Modern Python packaging

