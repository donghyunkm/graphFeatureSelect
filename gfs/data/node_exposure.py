"""Count exactly how many times the model sees each training node in one epoch, and build an
upsampled tabular dataset that reproduces those counts.

Why the counts are what they are: with ``trainmode: 0`` the training loss covers every node in the
sampled patch (``antelope.py:391``, ``train_mask`` is all-ones on the train graph), so a node
contributes one cross-entropy term for every ``NeighborLoader`` batch its patch lands in. Because
``batch.n_id`` is unique within a batch, "loss terms per node per epoch" == "number of batches the
node appears in". ``HalfHop`` slow nodes are dropped before the loss
(``antelope.py:406``), so they add nothing to the counts.

The dataloader is the real one -- this imports ``PyGAnnDataGraphDataModule`` from
``hemisphere_new`` and iterates the ``NeighborLoader`` it builds, so batch partitioning cannot
drift from training. Lightning calls ``train_dataloader()`` once and iterates it every epoch, so
this script does the same (one loader, ``--epochs`` passes over it).

Usage
-----
    python -m gfs.data.node_exposure --epochs 5
    python -m gfs.data.node_exposure --epochs 5 --materialize --genes 2,12,83,435,480

Outputs (in ``<data_root>/exposure/<tag>/``):
    node_exposure_counts.csv     per-cell counts, one column per epoch, plus mean
    node_exposure_summary.json   overall + per-subclass statistics
    upsampled_index.npy          int32 row index into the train cells, length sum(repeats)
    upsampled_train.h5ad         only with --materialize (chunked write, X = train X[index])
    tabular_val.h5ad / tabular_test.h5ad   only with --emit-valtest (not upsampled; see below)

val/test are NOT upsampled: ``validation_step`` / ``test_step`` score root nodes only
(``antelope.py:472,565``), so each val/test cell is seen exactly once per epoch.
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch

from gfs.data.hemisphere_new import PyGAnnDataGraphDataModule
from gfs.utils import get_paths


def count_exposure(datamodule, n_epochs, num_nodes, verbose=True):
    """Return an (n_epochs, num_nodes) int64 array of per-epoch, per-node loss-term counts.

    One loader is built and iterated ``n_epochs`` times, mirroring Lightning's behaviour when
    ``reload_dataloaders_every_n_epochs`` is unset.
    """
    loader = datamodule.train_dataloader()
    # NeighborLoaderMod's __iter__ recomputes subgraph_id per batch, which is expensive and does
    # not affect which nodes are in the patch. Iterate the wrapped NeighborLoader instead.
    inner = loader.neighborloader_obj
    n_batches = len(inner)

    counts = np.zeros((n_epochs, num_nodes), dtype=np.int64)
    for epoch in range(n_epochs):
        patch_sizes = []
        for batch in inner:
            n_id = batch.n_id.numpy()
            counts[epoch, n_id] += 1
            patch_sizes.append(n_id.size)
        if verbose:
            print(
                f"epoch {epoch}: {n_batches} batches, "
                f"patch size mean {np.mean(patch_sizes):.1f} (min {min(patch_sizes)}, max {max(patch_sizes)}), "
                f"total loss terms {counts[epoch].sum():,}, mean per node {counts[epoch].mean():.2f}",
                flush=True,
            )
    return counts


def summarize(counts, subclass, batch_size, n_hops, n_epochs):
    """Overall and per-subclass exposure statistics."""
    mean_counts = counts.mean(axis=0)
    n_nodes = mean_counts.size

    per_class = (
        pd
        .DataFrame({"subclass": subclass, "count_mean": mean_counts})
        .groupby("subclass", observed=True)["count_mean"]
        .agg(n_cells="size", exposure_mean="mean", exposure_std="std", exposure_total="sum")
        .sort_values("n_cells", ascending=False)
    )
    # relative over/under-exposure vs a graph-free tabular epoch, where every cell is seen once
    per_class["rel_to_uniform"] = per_class["exposure_mean"] / mean_counts.mean()

    summary = {
        "n_train_cells": int(n_nodes),
        "batch_size": int(batch_size),
        "n_hops": int(n_hops),
        "n_epochs_measured": int(n_epochs),
        "batches_per_epoch": int(np.ceil(n_nodes / batch_size)),
        "total_loss_terms_per_epoch": float(counts.sum(axis=1).mean()),
        "count_per_node": {
            "mean": float(mean_counts.mean()),
            "std": float(mean_counts.std()),
            "min": float(mean_counts.min()),
            "p1": float(np.percentile(mean_counts, 1)),
            "median": float(np.median(mean_counts)),
            "p99": float(np.percentile(mean_counts, 99)),
            "max": float(mean_counts.max()),
            "n_never_seen": int((mean_counts == 0).sum()),
        },
        "epoch_to_epoch_corr": (float(np.corrcoef(counts[0], counts[1])[0, 1]) if counts.shape[0] > 1 else None),
        "per_subclass": json.loads(per_class.reset_index().to_json(orient="records")),
    }
    return summary, per_class


def build_repeat_index(mean_counts, divisor, rng):
    """Turn per-node counts into a row index that repeats each cell ``round(count/divisor)`` times.

    Fractional remainders are resolved stochastically so the expected repeat count is exactly
    ``count/divisor`` rather than being systematically rounded.
    """
    scaled = mean_counts / divisor
    floor = np.floor(scaled)
    repeats = (floor + (rng.random(scaled.size) < (scaled - floor))).astype(np.int64)
    return np.repeat(np.arange(mean_counts.size, dtype=np.int64), repeats).astype(np.int32), repeats


def materialize_h5ad(path, adata_train, index, gene_cols, cell_type, classes, chunk_rows=200_000):
    """Write X = adata_train.X[index][:, gene_cols] to an h5ad without holding it in memory.

    X is streamed into an h5py dataset in row chunks; obs/var are written with anndata's element
    writer so the full repeated obs frame lands in the file without ever building a dense AnnData.
    """
    x = adata_train.X
    is_sparse = hasattr(x, "toarray")
    n_out, n_genes = index.size, len(gene_cols)

    # keep obs small: labels are what the downstream baseline needs, plus provenance
    cats = pd.Categorical(adata_train.obs[cell_type].astype(str).to_numpy()[index], categories=classes)
    obs = pd.DataFrame(
        {
            cell_type: cats,
            "label": cats.codes,
            "orig_index": index,
            "orig_cell_id": adata_train.obs_names.to_numpy()[index],
        },
        index=pd.RangeIndex(n_out).astype(str),
    )
    var = adata_train.var.iloc[gene_cols].copy()

    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"
        dset = f.create_dataset(
            "X",
            shape=(n_out, n_genes),
            dtype="float32",
            chunks=(min(chunk_rows, n_out), n_genes),
            compression="gzip",
        )
        for start in range(0, n_out, chunk_rows):
            rows = index[start : start + chunk_rows]
            block = x[rows, :].toarray() if is_sparse else x[rows, :]
            dset[start : start + rows.size, :] = np.asarray(block, dtype=np.float32)[:, gene_cols]
            print(f"  wrote rows {start:,}-{start + rows.size:,} of {n_out:,}", flush=True)

        ad.io.write_elem(f, "obs", obs)
        ad.io.write_elem(f, "var", var)
        ad.io.write_elem(f, "uns", {})
        ad.io.write_elem(f, "obsm", {})
        ad.io.write_elem(f, "varm", {})
        ad.io.write_elem(f, "obsp", {})
        ad.io.write_elem(f, "varp", {})
        ad.io.write_elem(f, "layers", {})
    return path


def emit_tabular_valtest(out_dir, path_valtest, cell_type, gene_cols, classes):
    """Write the val and test splits as plain tabular h5ads (no upsampling, root nodes only).

    ``classes`` is the train-derived class list. It is applied as the categorical ordering so the
    integer labels match the training set even for classes absent from val/test.
    """
    adata = ad.read_h5ad(path_valtest)
    written = []
    for split in ["val", "test"]:
        mask = adata.obs[f"{split}_mask"].to_numpy(dtype=bool)
        sub = adata[mask, :][:, gene_cols].to_memory() if adata.isbacked else adata[mask, :][:, gene_cols].copy()
        sub.obs[cell_type] = pd.Categorical(sub.obs[cell_type].astype(str), categories=classes)
        sub.obs["label"] = sub.obs[cell_type].cat.codes.to_numpy()
        assert (sub.obs["label"] >= 0).all(), f"{split} has a {cell_type} absent from the train classes"
        out = out_dir / f"tabular_{split}.h5ad"
        sub.write_h5ad(out)
        print(
            f"wrote {out} ({sub.shape[0]:,} cells x {sub.shape[1]} genes, "
            f"{sub.obs[cell_type].nunique()} of {len(classes)} {cell_type}s present)"
        )
        written.append(str(out))
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path-train", default="isocortex_train_final.h5ad", help="relative to data_root")
    p.add_argument("--path-valtest", default="isocortex_valtest_final.h5ad", help="relative to data_root")
    p.add_argument("--cell-type", default="subclass")
    p.add_argument("--batch-size", type=int, default=64, help="must match data.batch_size of the run being mirrored")
    p.add_argument("--n-hops", type=int, default=2, help="must match data.n_hops")
    p.add_argument("--rand-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5, help="epochs to measure; counts are averaged over them")
    p.add_argument(
        "--divisor",
        type=float,
        default=1.0,
        help="repeats = round(count / divisor). 1.0 reproduces the graph epoch exactly (~65x the "
        "train set). Use e.g. 8 for a smaller dataset with the same relative exposure.",
    )
    p.add_argument("--genes", default=None, help="comma-separated gene column indices to keep (default: all 500)")
    p.add_argument("--materialize", action="store_true", help="write upsampled_train.h5ad (large)")
    p.add_argument("--emit-valtest", action="store_true", help="also write plain tabular val/test h5ads")
    p.add_argument("--tag", default=None, help="output subdirectory name (default derived from settings)")
    p.add_argument(
        "--no-upsample",
        action="store_true",
        help="build a plain 1x tabular train (each cell seen exactly once); skips exposure counting "
        "and the upsampled repeat index. Ignores --divisor/--epochs. upsampled_train.h5ad then holds "
        "the raw train set sliced to the panel.",
    )
    args = p.parse_args()

    data_root = Path(get_paths()["data_root"])
    path_train = data_root / args.path_train
    path_valtest = data_root / args.path_valtest

    tag = args.tag or f"bs{args.batch_size}_hop{args.n_hops}_seed{args.rand_seed}"
    out_dir = data_root / "exposure" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    datamodule = PyGAnnDataGraphDataModule(
        data_dir=str(data_root),
        path_train=str(path_train),
        path_valtest=str(path_valtest),
        cell_type=args.cell_type,
        batch_size=args.batch_size,
        n_hops=args.n_hops,
        self_loops_only=False,
        rand_seed=args.rand_seed,
    )
    datamodule.setup("fit")
    adata_train = datamodule.dataset.adata_train
    num_nodes = datamodule.data_train.num_nodes
    print(f"train graph: {num_nodes:,} nodes, {datamodule.data_train.edge_index.size(1):,} edges")

    # ---- gene panel + train-derived label encoding (needed in every mode) ----
    gene_cols = [int(g) for g in args.genes.split(",")] if args.genes else list(range(adata_train.shape[1]))
    # label encoding must come from train: valtest is missing at least one class in this dataset,
    # so encoding per-file would silently shift the integer labels between splits.
    classes = datamodule.dataset.cell_type_list
    (out_dir / "classes.json").write_text(json.dumps({"cell_type": args.cell_type, "classes": classes}, indent=2))
    print(f"\n{len(classes)} classes (from train), written to {out_dir / 'classes.json'}")

    if args.no_upsample:
        # plain tabular baseline: each train cell appears exactly once, no exposure matching.
        index = np.arange(num_nodes, dtype=np.int32)
        repeats = np.ones(num_nodes, dtype=np.int64)
        np.save(out_dir / "upsampled_index.npy", index)
        np.save(out_dir / "upsampled_repeats.npy", repeats)
        print(f"\nno-upsample: plain 1x tabular train, {num_nodes:,} rows (exposure counting skipped)")
    else:
        # ---- exact per-node counts ----
        counts = count_exposure(datamodule, args.epochs, num_nodes)
        mean_counts = counts.mean(axis=0)
        subclass = adata_train.obs[args.cell_type].to_numpy()

        counts_df = pd.DataFrame({
            "cell_index": np.arange(num_nodes),
            "cell_id": adata_train.obs_names.to_numpy(),
            args.cell_type: subclass,
        })
        for e in range(args.epochs):
            counts_df[f"count_ep{e}"] = counts[e]
        counts_df["count_mean"] = mean_counts
        counts_path = out_dir / "node_exposure_counts.csv"
        counts_df.to_csv(counts_path, index=False)
        print(f"wrote {counts_path}")

        summary, per_class = summarize(counts, subclass, args.batch_size, args.n_hops, args.epochs)
        (out_dir / "node_exposure_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_dir / 'node_exposure_summary.json'}\n")
        print("per-node loss terms per epoch:", json.dumps(summary["count_per_node"], indent=2))
        print(
            "\nper-subclass exposure (rel_to_uniform = 1.0 means the graph neither over- nor under-samples this class):"
        )
        print(per_class.to_string(float_format=lambda v: f"{v:.3f}"))

        # ---- upsampled tabular dataset ----
        rng = np.random.default_rng(args.rand_seed)
        index, repeats = build_repeat_index(mean_counts, args.divisor, rng)
        np.save(out_dir / "upsampled_index.npy", index)
        np.save(out_dir / "upsampled_repeats.npy", repeats)
        print(
            f"\nupsampled dataset: {index.size:,} rows from {num_nodes:,} cells "
            f"({index.size / num_nodes:.2f}x, divisor {args.divisor})"
        )
        print(f"wrote {out_dir / 'upsampled_index.npy'} (int32 row index into the train cells)")

    if args.materialize:
        out = out_dir / "upsampled_train.h5ad"
        est_gb = index.size * len(gene_cols) * 4 / 1e9
        print(f"materializing {out} (~{est_gb:.1f} GB uncompressed, {len(gene_cols)} genes)...")
        materialize_h5ad(out, adata_train, index, gene_cols, args.cell_type, classes)
        print(f"wrote {out}")
    else:
        print(
            "skipped materialization (pass --materialize to write upsampled_train.h5ad); to use "
            "the index directly:\n"
            "    idx = np.load('upsampled_index.npy')\n"
            "    X_up, y_up = adata_train.X[idx][:, panel], labels[idx]"
        )

    if args.emit_valtest:
        emit_tabular_valtest(out_dir, path_valtest, args.cell_type, gene_cols, classes)


if __name__ == "__main__":
    main()
