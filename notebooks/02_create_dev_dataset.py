#!/usr/bin/env python
"""
Create a small dev dataset from a single well-balanced section.

Picks section index 600 (section 1199651094) from the training set:
- 3072 cells, 485 genes, 71 subclasses
- Balanced hemispheres: 1582 (False) / 1490 (True)

Merges metadata + gene expression into a single AnnData object per hemisphere.
Builds spatial neighbor graph using sklearn kneighbors_graph.
Saves to data/dev/.

Usage:
    python notebooks/02_create_dev_dataset.py
"""

from pathlib import Path

import anndata as ad

ad.settings.allow_write_nullable_strings = True
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "dev"
SECTION_IDX = 600
K_NEIGHBORS = 10
CELL_TYPE_COL = "AIT33_subclass"
SPATIAL_COORDS = ["center_x", "center_y"]


def load_section(idx: int) -> ad.AnnData:
    """Load and merge metadata + expression for a single section."""
    meta = ad.read_h5ad(RAW_DIR / f"638850_train/metadata/metadata_{idx:05d}.h5ad")
    sp = ad.read_h5ad(RAW_DIR / f"638850_train/sp-self/sp-self_{idx:05d}.h5ad")

    assert (sp.obs.index == meta.obs.index).all(), "Cell indices don't match"

    # Merge: use sp-self as base (has expression matrix), copy obs from metadata
    adata = sp.copy()
    adata.obs = meta.obs.copy()

    # Use gene_name as var index
    adata.var_names = adata.var["gene_name"].values
    adata.var_names_make_unique()

    return adata


def build_spatial_graph(adata: ad.AnnData, k: int, coords: list[str]) -> ad.AnnData:
    """Build k-nearest-neighbor spatial graph and store in obsp."""
    xy = adata.obs[coords].values.astype(np.float64)
    graph = kneighbors_graph(xy, n_neighbors=k, mode="connectivity", include_self=False)
    # Symmetrize: if A is neighbor of B, B is neighbor of A
    graph = ((graph + graph.T) > 0).astype(int)
    adata.obsp["spatial_connectivities"] = csr_matrix(graph)
    return adata


def split_and_save(adata: ad.AnnData, out_dir: Path):
    """Split by hemisphere and save each as a separate h5ad."""
    out_dir.mkdir(parents=True, exist_ok=True)

    section_id = adata.obs["section"].iloc[0]

    for reflected in [False, True]:
        hemi_label = "reflected" if reflected else "original"
        mask = adata.obs["is_reflected"] == reflected
        hemi = adata[mask].copy()

        # Build spatial graph per hemisphere (independent graphs)
        hemi = build_spatial_graph(hemi, k=K_NEIGHBORS, coords=SPATIAL_COORDS)

        path = out_dir / f"section_{section_id}_{hemi_label}.h5ad"
        hemi.write_h5ad(path)
        print(f"Saved {hemi_label} hemisphere: {hemi.shape[0]} cells, "
              f"{hemi.obs[CELL_TYPE_COL].nunique()} subclasses -> {path.name}")

    # Also save the full section (graph built on all cells together)
    full = build_spatial_graph(adata.copy(), k=K_NEIGHBORS, coords=SPATIAL_COORDS)
    full_path = out_dir / f"section_{section_id}_full.h5ad"
    full.write_h5ad(full_path)
    print(f"Saved full section: {full.shape[0]} cells -> {full_path.name}")


def main():
    print(f"Loading section {SECTION_IDX}...")
    adata = load_section(SECTION_IDX)
    print(f"  Section: {adata.obs['section'].iloc[0]}")
    print(f"  Cells: {adata.shape[0]}, Genes: {adata.shape[1]}")
    print(f"  Subclasses: {adata.obs[CELL_TYPE_COL].nunique()}")
    print(f"  Hemispheres: {adata.obs['is_reflected'].value_counts().to_dict()}")
    print()

    print("Splitting by hemisphere and building spatial graphs...")
    split_and_save(adata, OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
