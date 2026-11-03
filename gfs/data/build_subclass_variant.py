"""Build a relabeled 'abundant-style' subclass variant of the full isocortex data.

Recipe (mirrors the original ``0810_abudant_classes_only_top6`` build): keep a chosen set of
subclasses, collapse *every other* subclass (including the full data's pre-existing ``other``) into a
single ``other`` label. All cells are kept and everything except ``obs['subclass']`` is copied verbatim
from the full-data h5ads — including ``val_mask``/``test_mask``, so the train/valtest boundary and the
(fine-stratified) val/test split are reused unchanged. Because the full-data split is stratified at the
fine 22-subclass level (every fine subclass is 50/50 val:test within 1 cell), any collapse of them —
keep-set + ``other`` — inherits a ~50/50 val:test balance, with no cell changing split (no leakage).

Category order: kept subclasses sorted, then ``other`` last (matches the original abundant convention),
so ``.cat.codes`` are stable and ``other`` is the highest code.

Usage (from repo root, gfs env):
    python -m gfs.data.build_subclass_variant --family Glut --out 20260907_glut_classes
    python -m gfs.data.build_subclass_variant --family NN   --out 20260907_nn_classes
    python -m gfs.data.build_subclass_variant --family Gaba --out 20260907_gaba_classes
    # or an explicit keep-list:
    python -m gfs.data.build_subclass_variant --keep "052 Pvalb Gaba,053 Sst Gaba" --out mydir

Reads  <data_root>/isocortex_{train,valtest}_final.h5ad
Writes <data_root>/<out>/isocortex_{train,valtest}_final.h5ad
"""

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd

from gfs.utils import get_paths


def relabel(adata, keep, categories):
    s = adata.obs["subclass"].astype(str)
    new = np.where(s.isin(keep), s, "other")
    adata.obs["subclass"] = pd.Categorical(new, categories=categories)
    return adata


def balance_report(valtest, categories):
    obs = valtest.obs
    val, test = obs[obs["val_mask"]], obs[obs["test_mask"]]
    vc, tc = val["subclass"].value_counts(), test["subclass"].value_counts()
    df = pd.DataFrame({"val": vc, "test": tc}).reindex(categories).fillna(0).astype(int)
    df["total"] = df["val"] + df["test"]
    df["diff"] = (df["val"] - df["test"]).abs()
    return df


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--family", choices=["Glut", "NN", "Gaba"], help="keep all subclasses with this suffix")
    g.add_argument("--keep", help="comma-separated explicit subclass names to keep")
    ap.add_argument("--out", required=True, help="output subdir under data_root")
    ap.add_argument(
        "--src",
        default="",
        help="source subdir under data_root to read isocortex_{train,valtest}_final.h5ad from "
        "(default: data_root itself, i.e. the 22-label full data). Use e.g. '0914_all_subclasses' "
        "to build variants from the enriched 34-label label set.",
    )
    args = ap.parse_args()

    data_root = get_paths()["data_root"]
    src = f"{data_root}{args.src.rstrip('/') + '/' if args.src else ''}"
    train = ad.read_h5ad(f"{src}isocortex_train_final.h5ad")
    valtest = ad.read_h5ad(f"{src}isocortex_valtest_final.h5ad")

    all_cats = list(train.obs["subclass"].cat.categories)
    if args.family:
        keep = [c for c in all_cats if c.split()[-1] == args.family]
    else:
        keep = [c.strip() for c in args.keep.split(",")]
    unknown = set(keep) - set(all_cats)
    assert not unknown, f"keep-set has subclasses absent from the data: {unknown}"
    assert "other" not in keep, "'other' is the collapse target; do not put it in --keep"

    categories = [*sorted(keep), "other"]
    print(f"keep ({len(keep)}): {keep}")
    print(f"-> {len(categories)} labels (kept + other), order: {categories}")

    relabel(train, keep, categories)
    relabel(valtest, keep, categories)

    out_dir = f"{data_root}{args.out}"
    os.makedirs(out_dir, exist_ok=True)
    train.write_h5ad(f"{out_dir}/isocortex_train_final.h5ad")
    valtest.write_h5ad(f"{out_dir}/isocortex_valtest_final.h5ad")

    # Verification: split balance + no cells lost/moved.
    df = balance_report(valtest, categories)
    print(f"\ntrain cells: {train.n_obs} | valtest cells: {valtest.n_obs}")
    print("per-class val/test balance (val_frac ~0.5 => stratified, diff<=few => balanced):")
    df["val_frac"] = (df["val"] / df["total"]).round(4)
    print(df[["val", "test", "total", "diff", "val_frac"]])
    print(f"\nmax |val-test| across classes: {int(df['diff'].max())}")
    print(f"train subclass counts:\n{train.obs['subclass'].value_counts()}")
    print(f"\nwrote {out_dir}/isocortex_{{train,valtest}}_final.h5ad")


if __name__ == "__main__":
    main()
