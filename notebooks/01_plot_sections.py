#!/usr/bin/env python
"""
Plot CCF coordinates colored by cell type (AIT33_subclass) for multiple sections.

This script:
- Loads the 638850-metadata.h5ad file
- Selects ~10 random sections
- Plots xy, yz, and zx projections colored by AIT33_subclass
- Saves visualizations to ../outputs/

Usage:
    python 01_plot_sections.py --data-file ../data/raw/638850-metadata.h5ad
    python 01_plot_sections.py --n-sections 10 --coord-type reflected_scaled
"""

import argparse
from pathlib import Path
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def load_metadata(data_path: Path) -> ad.AnnData:
    """Load metadata file."""
    print(f"Loading metadata from {data_path}...")
    adata = ad.read_h5ad(data_path)
    print(f"Loaded: {adata.shape}")
    print(f"Sections available: {adata.obs['section'].nunique()}")
    return adata


def select_random_sections(adata: ad.AnnData, n_sections: int = 10, seed: int = 42) -> list:
    """Select random sections with sufficient cells."""
    np.random.seed(seed)

    # Get section counts
    section_counts = adata.obs['section'].value_counts()
    print(f"\nSection statistics:")
    print(f"  Total sections: {len(section_counts)}")
    print(f"  Cells per section: {section_counts.min()} - {section_counts.max()}")

    # Select sections with enough cells (> 1000)
    valid_sections = section_counts[section_counts > 1000].index.tolist()

    # Randomly select n_sections
    if len(valid_sections) < n_sections:
        print(f"Warning: Only {len(valid_sections)} sections with >1000 cells")
        n_sections = len(valid_sections)

    selected = np.random.choice(valid_sections, size=n_sections, replace=False)
    selected = sorted(selected)

    print(f"\nSelected {len(selected)} sections:")
    for sec in selected:
        n_cells = section_counts[sec]
        print(f"  Section {sec}: {n_cells:,} cells")

    return selected


def get_color_map(adata: ad.AnnData) -> dict:
    """Create consistent color mapping for all cell types."""
    cell_types = sorted(adata.obs['AIT33_subclass'].dropna().unique())
    n_types = len(cell_types)

    # Use tab20 for up to 20 types, hsv for more
    if n_types <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_types))
    else:
        colors = plt.cm.hsv(np.linspace(0, 0.95, n_types))

    return dict(zip(cell_types, colors))


def plot_projection(ax, section_data: pd.DataFrame, x_col: str, y_col: str,
                   color_map: dict, title: str, subsample_factor: int = 10,
                   reflected: bool = False):
    """
    Plot a single 2D projection.

    Args:
        ax: Matplotlib axis
        section_data: DataFrame with section data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_map: Dictionary mapping cell types to colors
        title: Plot title
        subsample_factor: Factor by which to subsample cells (default: 10)
        reflected: If True, plot reflected cells; if False, plot non-reflected cells
    """
    # Filter for reflected or non-reflected cells
    section_data = section_data[section_data['is_reflected'] == reflected].copy()

    # Subsample data
    if len(section_data) > subsample_factor:
        section_data = section_data.iloc[::subsample_factor]

    # Plot each cell type
    for cell_type, color in color_map.items():
        mask = section_data['AIT33_subclass'] == cell_type
        subset = section_data[mask]

        if len(subset) > 0:
            ax.scatter(
                subset[x_col],
                subset[y_col],
                c=[color],
                s=1,
                alpha=0.6,
                rasterized=True
            )

    # Handle NaN values
    nan_mask = section_data['AIT33_subclass'].isna()
    if nan_mask.sum() > 0:
        ax.scatter(
            section_data[nan_mask][x_col],
            section_data[nan_mask][y_col],
            c='lightgray',
            s=1,
            alpha=0.3,
            rasterized=True
        )

    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linewidth=0.5)


def plot_section_3views(adata: ad.AnnData, section_id: str, coord_type: str = "CCF",
                        color_map: dict = None, output_dir: Path = None, dpi: int = 150,
                        subsample_factor: int = 10, reflected: bool = False):
    """
    Plot three 2D projections (xy, yz, zx) for a single section.

    Args:
        adata: AnnData object
        section_id: Section identifier
        coord_type: Type of coordinates - "CCF", "reflected", or "reflected_scaled"
        color_map: Dictionary mapping cell types to colors
        output_dir: Directory to save plot
        dpi: Resolution for saved figure
        subsample_factor: Factor by which to subsample cells
        reflected: If True, plot reflected cells; if False, plot non-reflected cells
    """
    # Filter data for this section
    section_data = adata[adata.obs['section'] == section_id].obs

    # Get stats before filtering
    total_cells = len(section_data)
    filtered_cells = (section_data['is_reflected'] == reflected).sum()
    plotted_cells = filtered_cells // subsample_factor

    # Select coordinate columns
    if coord_type == "CCF":
        x_col, y_col, z_col = "x_CCF", "y_CCF", "z_CCF"
        suffix = "CCF"
    elif coord_type == "reflected":
        x_col, y_col, z_col = "x_CCF_reflected", "y_CCF_reflected", "z_CCF_reflected"
        suffix = "reflected"
    elif coord_type == "reflected_scaled":
        x_col = "x_CCF_reflected_scaled"
        y_col = "y_CCF_reflected_scaled"
        z_col = "z_CCF_reflected_scaled"
        suffix = "scaled"
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot XY projection
    plot_projection(axes[0], section_data, x_col, y_col, color_map,
                   f"XY view - Section {section_id}", subsample_factor, reflected)

    # Plot YZ projection
    plot_projection(axes[1], section_data, y_col, z_col, color_map,
                   f"YZ view - Section {section_id}", subsample_factor, reflected)

    # Plot ZX projection
    plot_projection(axes[2], section_data, z_col, x_col, color_map,
                   f"ZX view - Section {section_id}", subsample_factor, reflected)

    reflected_label = "reflected" if reflected else "non-reflected"
    fig.suptitle(f"Section {section_id} - {plotted_cells:,} cells plotted "
                f"({reflected_label}, 1/{subsample_factor} sample) ({suffix} coordinates)",
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    if output_dir:
        reflected_suffix = "reflected" if reflected else "nonreflected"
        output_path = output_dir / f"section_{section_id}_{coord_type}_{reflected_suffix}_3views.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_path.name} ({filtered_cells:,} → {plotted_cells:,} cells)")

    plt.close(fig)


def create_summary_plot(adata: ad.AnnData, sections: list, coord_type: str = "CCF",
                       color_map: dict = None, output_dir: Path = None, dpi: int = 150,
                       subsample_factor: int = 10, reflected: bool = False):
    """Create a summary plot with XY view of all sections in a grid."""
    n_sections = len(sections)
    n_cols = min(4, n_sections)
    n_rows = int(np.ceil(n_sections / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_sections == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Select coordinate columns
    if coord_type == "CCF":
        x_col, y_col = "x_CCF", "y_CCF"
        suffix = "CCF"
    elif coord_type == "reflected":
        x_col, y_col = "x_CCF_reflected", "y_CCF_reflected"
        suffix = "reflected"
    elif coord_type == "reflected_scaled":
        x_col = "x_CCF_reflected_scaled"
        y_col = "y_CCF_reflected_scaled"
        suffix = "scaled"

    for idx, section_id in enumerate(sections):
        ax = axes[idx]
        section_data = adata[adata.obs['section'] == section_id].obs

        # Filter for reflected or non-reflected cells
        section_data = section_data[section_data['is_reflected'] == reflected].copy()

        # Subsample data
        if len(section_data) > subsample_factor:
            section_data = section_data.iloc[::subsample_factor]

        # Plot each cell type
        for cell_type, color in color_map.items():
            mask = section_data['AIT33_subclass'] == cell_type
            subset = section_data[mask]

            if len(subset) > 0:
                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    c=[color],
                    s=0.3,
                    alpha=0.6,
                    rasterized=True
                )

        ax.set_title(f"Section {section_id}\n{len(section_data):,} cells", fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7)

    # Hide extra subplots
    for idx in range(n_sections, len(axes)):
        axes[idx].axis('off')

    reflected_label = "reflected" if reflected else "non-reflected"
    fig.suptitle(f"Brain 638850 - XY Spatial Distribution by AIT33_subclass\n"
                f"({reflected_label}, 1/{subsample_factor} sample, {suffix} coordinates)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        reflected_suffix = "reflected" if reflected else "nonreflected"
        output_path = output_dir / f"summary_all_sections_{coord_type}_{reflected_suffix}_xy.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nSummary plot saved: {output_path.name}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CCF coordinates colored by cell type")
    parser.add_argument(
        "--data-file",
        type=str,
        default="../data/638850-metadata.h5ad",
        help="Path to metadata h5ad file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../outputs",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--n-sections",
        type=int,
        default=10,
        help="Number of sections to plot"
    )
    parser.add_argument(
        "--coord-type",
        type=str,
        default="reflected_scaled",
        choices=["CCF", "reflected", "reflected_scaled"],
        help="Type of coordinates to use"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for saved figures"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for section selection"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=10,
        help="Subsample factor (plot every Nth cell, default: 10)"
    )
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    data_path = (script_dir / args.data_file).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    print("="*60)
    print("Section Plotting Script - 3 View Projections")
    print("="*60)
    print(f"Data file: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Coordinate type: {args.coord_type}")

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    adata = load_metadata(data_path)

    # Create consistent color map for all cell types
    print("\nCreating color map for cell types...")
    color_map = get_color_map(adata)
    print(f"Color map created for {len(color_map)} cell types")

    # Select sections
    sections = select_random_sections(adata, args.n_sections, args.seed)

    # Plot individual sections (3 views each) - Non-reflected cells
    print(f"\nPlotting 3-view projections for each section (non-reflected cells)...")
    print(f"Filters: is_reflected=False, subsample=1/{args.subsample}")
    for section_id in sections:
        print(f"Section {section_id}...")
        plot_section_3views(adata, section_id, args.coord_type, color_map,
                           output_dir, args.dpi, args.subsample, reflected=False)

    # Create summary plot (XY view only) - Non-reflected cells
    print(f"\nCreating summary plot (XY view, non-reflected cells)...")
    create_summary_plot(adata, sections, args.coord_type, color_map,
                       output_dir, args.dpi, args.subsample, reflected=False)

    # Plot individual sections (3 views each) - Reflected cells
    print(f"\nPlotting 3-view projections for each section (reflected cells)...")
    print(f"Filters: is_reflected=True, subsample=1/{args.subsample}")
    for section_id in sections:
        print(f"Section {section_id}...")
        plot_section_3views(adata, section_id, args.coord_type, color_map,
                           output_dir, args.dpi, args.subsample, reflected=True)

    # Create summary plot (XY view only) - Reflected cells
    print(f"\nCreating summary plot (XY view, reflected cells)...")
    create_summary_plot(adata, sections, args.coord_type, color_map,
                       output_dir, args.dpi, args.subsample, reflected=True)

    print("\n" + "="*60)
    print("Plotting complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
