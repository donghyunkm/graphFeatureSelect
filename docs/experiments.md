# Running Experiments

## Quick Start

```bash
# Activate environment
conda activate gfs 

# Run with default configuration
python gfs/trainers/antelope.py

## Configuration System

GFSNet uses hydra for hierarchical configuration management.

### Config Structure

Base configuration: `gfs/configs/antelope.yaml`

### Overriding Parameters

```bash
# Change dataset
python gfs/trainers/antelope.py \
  data.file_names='["custom_dataset.h5ad"]'

# Modify model architecture
python gfs/trainers/antelope.py \
  model.hid_ch=64 \
  model.local_layers=3 \
  model.dropout=0.3

# Try different GNN backbone
python gfs/trainers/antelope.py \
  model.gnn="sage"

# Use focal loss for imbalanced data
python gfs/trainers/antelope.py \
  model.focal_loss=true

# Quick test run
python gfs/trainers/antelope.py \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=5 \
  trainer.max_epochs=5
```

## Training Modes

### Standard Training (`trainmode=0`)

Computes loss and metrics on all nodes in each batch (root + neighbors):

```bash
python gfs/trainers/antelope.py model.trainmode=0
```

### Root-Only Training (`trainmode=1`)

Computes loss and metrics only on root nodes:

```bash
python gfs/trainers/antelope.py model.trainmode=1
```

## Output Structure

### Logs

Training logs saved to: `data/logs/<experiment_name>/`

Contents:
- TensorBoard events
- `selections.csv` - Gene selections per epoch
- Metric logs

View with TensorBoard:
```bash
tensorboard --logdir data/logs/
```

### Checkpoints

Model checkpoints saved to: `data/checkpoints/<experiment_name>/`

Format: `{epoch}-{val_overall_acc:.2f}.ckpt`

Only the best checkpoint (by validation accuracy) is kept.

### Predictions

Test predictions saved to: `data/logs/<experiment_name>/test_pred.pt`

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

Track which genes are selected over training:

```python
import pandas as pd

selections = pd.read_csv('data/logs/<experiment>/selections.csv')

# View final selections
final_epoch = selections.iloc[-1]
selected_genes = [final_epoch[f'sel_{i}'] for i in range(10)]
selection_probs = [final_epoch[f'prob_{i}'] for i in range(10)]
```

### Performance Metrics

Extract from TensorBoard logs or final test output:

- `test_overall_acc` - Weighted accuracy
- `test_macro_acc` - Macro-averaged accuracy
- `test_micro_acc` - Micro-averaged accuracy
- `test_f1_overall` - Weighted F1
- `test_f1_macro` - Macro-averaged F1
- `test_f1_micro` - Micro-averaged F1

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

conda activate gfs
python gfs/trainers/antelope.py data.prefix="experiment_name"
```
## Model Variants

### Run Different Models

```bash
# STG variant
python gfs/trainers/antelope_stg.py
```
Each variant should have its own config file in `gfs/configs/`.

## Troubleshooting

### Out of Memory

Reduce batch size or limit batches:
```bash
python gfs/trainers/antelope.py \
  data.batch_size=32 \
  trainer.limit_train_batches=500
```

### Slow Training

- Use smaller validation set: `trainer.limit_val_batches=50`
- Reduce GNN depth: `model.local_layers=1`
- Decrease neighborhood size: `data.n_hops=1`

### Poor Convergence

- Check gene selection logs for stability
- Try different temperature schedule: `model.tautype="constant"`
- Increase training epochs: `trainer.max_epochs=1000`
- Adjust learning rate: `trainer.lr=0.0001`

### Feature Selection Not Working

- Monitor `selections.csv` - selections should stabilize over time
- Ensure temperature annealing is enabled: `model.tautype="exp"`
- Check that `n_select` matches desired panel size
- Verify sufficient training epochs (500+)
