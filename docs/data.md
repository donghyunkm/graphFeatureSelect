# Data: Preprocessing, Transformation, and Normalized Format

## Raw data

Source: Allen Institute brain 638850 spatial transcriptomics.

**Location:** `data/raw/`

Two file types per section, stored as separate h5ad files:
- `metadata_NNNNN.h5ad` — cell annotations in `.obs`, no expression data
- `sp-self_NNNNN.h5ad` — gene expression in `.X` (n_cells x 485 genes), no `.obs`

Files share cell indices and can be joined on `.obs.index`.

Already split into `638850_train/` (1585 sections), `638850_val/` (89), `638850_test/` (89).

### Key metadata columns

| Column | Description |
|--------|-------------|
| `AIT33_subclass` | Cell type label (413 unique across whole brain) |
| `AIT33_class` | Coarse cell type (43 unique) |
| `AIT33_supertype` | Fine cell type (1379 unique) |
| `section` | Coronal section ID (57 unique) |
| `is_reflected` | Boolean: which hemisphere |
| `center_x`, `center_y` | Spatial coordinates within section |
| `x_CCF`, `y_CCF`, `z_CCF` | Allen CCF coordinates |
| `*_CCF_reflected`, `*_CCF_reflected_scaled` | Aligned/scaled coordinates |

### Gene expression

- 485 genes (chromatin accessibility features)
- Float32, range [0, ~13], mean ~2.3
- ~70% zeros (sparse)
- Gene names in `.var['gene_name']` (e.g. Oprk1, St18, Pou3f3)
- No blank genes in sp-self files

## Preprocessing pipeline

Script: `notebooks/02_create_dev_dataset.py`

Steps:
1. **Load** metadata + expression for a section, join on cell index
2. **Set gene names** as var index (from `var['gene_name']`)
3. **Split by hemisphere** using `is_reflected` column
4. **Build spatial graph** per hemisphere: k-nearest-neighbor graph on `[center_x, center_y]`, symmetrized, stored in `obsp['spatial_connectivities']`
5. **Save** as independent h5ad files (one per hemisphere)

### Graph construction

- Algorithm: `sklearn.neighbors.kneighbors_graph(k=10, mode='connectivity')`
- Symmetrized: if A is neighbor of B, B is neighbor of A
- Per-hemisphere: no edges cross between hemispheres
- Stored as: `obsp['spatial_connectivities']` (sparse int matrix)

## Normalized format

Each h5ad file consumed by the training pipeline has:

```
adata.X                          # (n_cells, n_genes) float32, gene expression
adata.obs['AIT33_subclass']      # cell type labels (categorical)
adata.obs['center_x']            # spatial x coordinate
adata.obs['center_y']            # spatial y coordinate
adata.obs['section']             # section ID
adata.obs['is_reflected']        # hemisphere indicator
adata.obsp['spatial_connectivities']  # (n_cells, n_cells) sparse adjacency
adata.var_names                  # gene names as index
```

**Invariants:**
- All cells have a label in the chosen cell type column
- Spatial graph is symmetric, binary, no self-loops
- Gene expression is float32, non-negative
- Each file is one independent graph (no cross-file edges)

## Dev dataset

**Location:** `data/dev/`

Created from section 600 (section ID 1199651094), chosen for balanced hemispheres.

| File | Cells | Genes | Subclasses | Edges | Use |
|------|-------|-------|------------|-------|-----|
| `section_1199651094_original.h5ad` | 1,582 | 485 | 46 | 18,454 | Train/val |
| `section_1199651094_reflected.h5ad` | 1,490 | 485 | 63 | 17,420 | Test |
| `section_1199651094_full.h5ad` | 3,072 | 485 | 71 | 35,848 | Transductive baseline |

## Normalization

Expression values are **log1p(CPM)** — confirmed by inspection of upstream processing.

- Range: [0, ~13], mean ~2.3, ~70% zeros
- No additional normalization is applied before feeding to the model
- Reconstruction target: predict log1p(CPM) values directly
