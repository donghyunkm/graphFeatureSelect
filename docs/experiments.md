# Running Experiments

## Quick Start

```bash
# Activate environment
conda activate gfsnet

# Run with default configuration (Gumbel + GAT + classification)
python -m gfs.trainers.train

# Specify feature selection and backbone
python -m gfs.trainers.train \
  backbone=gat feature_selection=stg task=classification \
  n_select=10 lam=0.1
```

## Configuration System

GFSNet uses Hydra for hierarchical, composable configuration management.

### Config Structure

Base configuration: `src/gfs/conf/config.yaml`

Config groups (independently composable):
- `backbone/` -- GNN architecture (`gat`, `sage`, `gcn`)
- `feature_selection/` -- gene selection method (`gumbel`, `stg`, `scgist`)
- `task/` -- prediction head (`classification`, `reconstruction`)
- `data/` -- dataset and splitting (`hemisphere`)
- `trainer/` -- training hyperparameters (`default`)
- `logging/` -- log timing and destinations (`default`)

Global flags (set directly on the command line):
- `n_select=10` -- number of genes to select
- `trainmode=0` -- 0=all nodes, 1=seed only
- `lam=0.1` -- regularization weight for feature selection
- `halfhop=false` -- HalfHop graph augmentation

### Overriding Parameters

```bash
# Modify backbone architecture
python -m gfs.trainers.train \
  backbone.hid_ch=64 \
  backbone.n_layers=3 \
  backbone.dropout=0.3

# Try different GNN backbone
python -m gfs.trainers.train backbone=sage

# Use STG feature selection with custom sigma
python -m gfs.trainers.train \
  feature_selection=stg \
  feature_selection.sigma=1.0 \
  lam=0.05

# Use scGist feature selection
python -m gfs.trainers.train \
  feature_selection=scgist \
  feature_selection.l1=0.2

# Use reconstruction task head
python -m gfs.trainers.train task=reconstruction

# Quick test run
python -m gfs.trainers.train \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=5 \
  trainer.max_epochs=5
```

## Training Modes

### Standard Training (`trainmode=0`)

Computes loss and metrics on all nodes in each batch (root + neighbors):

```bash
python -m gfs.trainers.train trainmode=0
```

### Root-Only Training (`trainmode=1`)

Computes loss and metrics only on seed (root) nodes:

```bash
python -m gfs.trainers.train trainmode=1
```

## Output Structure

### Logs

Training logs saved to: `logs/<experiment_name>/`

Contents:
- TensorBoard events
- Metric logs

View with TensorBoard:
```bash
tensorboard --logdir logs/
```

### Checkpoints

Model checkpoints saved to: `checkpoints/<experiment_name>/`

Format: `{epoch}-{val_acc:.2f}.ckpt`

Only the best checkpoint (by validation accuracy) is kept.

### Predictions

Test predictions saved to: `logs/<experiment_name>/test_pred.pt`

Contents:
```python
{
  'predictions': torch.Tensor,  # Shape: (n_test_samples, n_classes)
  'labels': torch.Tensor         # Shape: (n_test_samples,)
}
```

Load predictions:
```python
import torch
results = torch.load('path/to/test_pred.pt')
predictions = results['predictions']
labels = results['labels']
```

## Analyzing Results

### Gene Selections

Inspect selected genes at eval time via the feature selector:

```python
model.feature_selector.selected_indices()  # tensor of gene indices
```

For Gumbel selectors, per-slot probabilities are available via:

```python
indices, probs = model.feature_selector.get_mask_indices()
```

### Performance Metrics

Extract from TensorBoard logs or final test output:

- `test_acc` -- Weighted accuracy
- `test_macro_acc` -- Macro-averaged accuracy
- `test_f1_macro` -- Macro-averaged F1
- `test_loss` -- Task loss

## SLURM Job Submission

### Basic Job

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH -t 7000
#SBATCH -J gfs_experiment

conda activate gfsnet
python -m gfs.trainers.train \
  expname="experiment_name" \
  backbone=gat feature_selection=stg task=classification \
  n_select=20 lam=0.1
```

## Feature Selection Variants

### Gumbel Softmax (default)

```bash
python -m gfs.trainers.train feature_selection=gumbel
```

Uses temperature annealing (`tautype=exp`) by default. Override with:
```bash
python -m gfs.trainers.train feature_selection=gumbel feature_selection.tautype=constant
```

### Stochastic Gates

```bash
python -m gfs.trainers.train feature_selection=stg lam=0.1
```

Key parameters: `feature_selection.sigma` (noise scale) and `lam` (sparsity weight).

### scGist

```bash
python -m gfs.trainers.train feature_selection=scgist lam=0.1
```

Key parameter: `feature_selection.l1` (binary + panel size regularization weight).

## Troubleshooting

### Out of Memory

Reduce batch size or limit batches:
```bash
python -m gfs.trainers.train \
  data.batch_size=32 \
  trainer.limit_train_batches=500
```

### Slow Training

- Use smaller validation set: `trainer.limit_val_batches=50`
- Reduce GNN depth: `backbone.n_layers=1`
- Decrease neighborhood size: `data.n_hops=1`

### Poor Convergence

- For Gumbel: check that temperature annealing is enabled (`feature_selection.tautype=exp`)
- For STG/scGist: adjust `lam` to balance classification vs sparsity
- Increase training epochs: `trainer.max_epochs=1000`
- Adjust learning rate: `trainer.lr=0.0001`

### Feature Selection Not Working

- Ensure `n_select` matches desired panel size
- For Gumbel: ensure sufficient epochs (500+) for temperature to anneal
- For STG: try different `sigma` values (0.5 default, lower = sharper gates)
- For scGist: increase `feature_selection.l1` if weights are not binarizing
