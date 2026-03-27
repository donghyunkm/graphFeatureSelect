# Development Guide

## Environment Setup

### Option 1: Using uv (Recommended for local development)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch with CUDA support (CUDA 12.1)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install PyG dependencies
uv pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install package in editable mode with all dependencies
uv pip install -e ".[dev,mldep,data]"

# Generate/update lock file
uv lock

# Setup pre-commit hooks
pre-commit install
```

### Option 2: Using conda + uv (Recommended for GPU environments)

```bash
# Install miniconda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create conda environment with Python 3.12
conda create -n gfsnet python=3.12 -y
conda activate gfsnet

# Install PyTorch with CUDA support (CUDA 12.1)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install PyG dependencies
uv pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install package in editable mode with all dependencies
# Dependency groups: dev (development tools), mldep (ML/DL libraries), data (data science/bioinformatics)
uv pip install -e ".[dev,mldep,data]"

# Generate/update lock file
uv lock

# Setup pre-commit hooks
pre-commit install

# Verify GPU installation
python tests/test_install.py
```

### Dependency Groups

The project dependencies are organized into three optional groups in `pyproject.toml`:

- **dev**: Development tools (build, pre-commit, ruff, ipywidgets, visualization, CLI tools)
- **mldep**: Machine learning dependencies (torch, lightning, torch-geometric, hydra, tensorboard)
- **data**: Data science and bioinformatics (numpy, scipy, pandas, scikit-learn, anndata, scanpy, ipython, jupyterlab)

Install specific groups as needed:
```bash
uv pip install -e ".[dev]"           # Development only
uv pip install -e ".[mldep,data]"    # ML and data science only
uv pip install -e ".[dev,mldep,data]" # All dependencies
```

## Build System

- **Build backend**: Hatchling (modern, standards-compliant build system)
- **Package manager**: uv (preferred), also compatible with pip and conda
- **Python requirement**: >=3.10 (recommended: 3.12)
- **Dependency locking**: uv.lock file for reproducible builds
- Package structure defined in `pyproject.toml` using PEP 621 standards

### Installed Versions
- **PyTorch**: 2.5.1+cu121
- **PyTorch Geometric**: 2.7.0
- **CUDA**: 12.1
- **Python**: 3.12.12

## Code Style

### Linting & Formatting

Uses `ruff` for both linting and formatting:

- **Line length**: 120 characters
- **Enabled rules**:
  - `I` - isort (import sorting)
  - `N` - pep8-naming
  - `NPY` - NumPy-specific conventions
  - `RUF` - Ruff-specific rules

### Pre-commit Hooks

Pre-commit hooks are configured in `.pre-commit-config.yaml` and run automatically before each commit:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### Ignored Rules

The following rules are intentionally ignored:
- `N801` - Class names that ignore CapWords convention
- `N802` - Function names that are not all lowercase
- `N803` - Argument names (fine for matrices)
- `N806` - Variable names (fine for matrices)
- `N816` - CapWords for function names
- `NPY002` - `np.random` calls (for reproducibility)
- `N812` - Lowercase imported as non-lowercase

## Development Workflow

### Running Tests

```bash
# Run all tests (90 integration tests)
conda run -n gfsnet python -m pytest

# Run specific test file
conda run -n gfsnet python -m pytest tests/test_data_pipeline.py -v

# Run with coverage
conda run -n gfsnet python -m pytest --cov=gfs
```

### Building Package

```bash
# Build distribution
python -m build

# The wheel and sdist will be in dist/
```

### Debugging Tips

- Set `limit_train_batches` and `limit_val_batches` to small values for quick iteration
- Check gene selection logs to verify feature selection is working
- Use `trainmode=0` for standard training, `trainmode=1` to backprop only through root nodes
- The HalfHop transform can be disabled by setting `halfhop: false`
- Print statements in model forward pass show tensor shapes (search for "SAMPLE", "KHOT")

## Tech Stack

- **Deep Learning**: PyTorch, PyTorch Geometric, Lightning
- **Configuration**: Hydra for experiment configuration management
- **Data**: AnnData/h5ad format for spatial transcriptomics data
- **Metrics**: scikit-learn, torchmetrics
- **Visualization**: tensorboard, matplotlib, seaborn
- **Development**: ruff, pre-commit hooks
