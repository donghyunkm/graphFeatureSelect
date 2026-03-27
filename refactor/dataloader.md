# Dataloader: Subgraph Sampling

## Overview

The model operates on local spatial patches (subgraphs), not the full graph. Each training example is a seed node plus its k-hop neighborhood, sampled by PyG's `NeighborLoader`.

## Pipeline

```
h5ad file
    |
    v
PyGAnnData          Convert AnnData -> PyG Data object
    |               (gene_exp, xyz, y, edge_index, masks)
    v
NeighborLoader      Sample k-hop neighborhoods around seed nodes
    |               (PyG built-in, returns batched subgraphs)
    v
Batch               Ready for model forward pass
```

## NeighborLoader configuration

```python
NeighborLoader(
    data,
    num_neighbors=[-1] * n_hops,  # all neighbors at each hop
    batch_size=64,                 # seed nodes per batch
    input_nodes=mask,              # train/val/test mask
    shuffle=True,                  # for training
    num_workers=8,
)
```

- `num_neighbors=[-1]`: sample ALL neighbors (not a random subset)
- `n_hops=2`: 2-hop neighborhoods (default)
- Each batch contains ~64 seed nodes + their 2-hop neighborhoods
- A batch typically has a few hundred to a few thousand nodes total

## Subgraph ID assignment

`subgraph_id` maps every node to the seed node whose neighborhood it belongs to. When not explicitly provided, it defaults to all-zeros (single subgraph).

## What the model receives per batch

```python
batch.gene_exp        # (n_batch_nodes, n_genes) -- gene expression
batch.xyz             # (n_batch_nodes, n_coords) -- spatial coordinates
batch.y               # (n_batch_nodes,) -- cell type labels
batch.edge_index      # (2, n_edges) -- graph connectivity within batch
batch.train_mask      # (n_batch_nodes,) -- which nodes are in training set
batch.subgraph_id     # (n_batch_nodes,) -- maps node -> seed node index
batch.n_id            # (n_batch_nodes,) -- original node IDs in full graph
batch.input_id        # (n_seed_nodes,) -- original IDs of seed nodes
```

## Seed node identification

Seed (central) nodes are the first `batch.input_id.size(0)` entries in the batch:
```python
n_seed = batch.input_id.size(0)
seed_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
seed_mask[:n_seed] = True
```

This is used in val/test to restrict metrics to seed nodes only.

## Feature masking flow

1. `batch.gene_exp` -- raw gene expression per node
2. Feature selector applies mask: `masked_exp = mask * gene_exp`
   - All nodes in same subgraph get same mask (via `subgraph_id`)
   - Different subgraphs may get different sampled masks (training)
3. Masked expression enters GNN for message passing
4. Task head produces predictions from GNN output embeddings

## Train vs eval behavior

| Aspect | Training | Validation/Test |
|--------|----------|-----------------|
| Seed nodes | `train_mask` | `val_mask` / `test_mask` |
| Shuffle | Yes | No |
| Feature mask | Soft (continuous) | Hard (binary) |
| Loss nodes | All train nodes (`trainmode=0`) or seed only (`trainmode=1`) | Seed nodes only |
| Metric nodes | Same as loss | Seed nodes only |

## Data format: separate attributes

Gene expression and spatial coordinates are stored as separate attributes on the PyG Data object:
```python
data.gene_exp = torch.tensor(adata.X)           # (n_nodes, n_genes)
data.xyz = torch.tensor(coords)                  # (n_nodes, n_coords)
data.y = torch.tensor(labels)                    # (n_nodes,)
```

This avoids index slicing and keeps the data model clean.
