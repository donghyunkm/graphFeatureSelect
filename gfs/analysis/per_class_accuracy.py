"""Recompute per-class accuracy (recall) from `test_pred.pt` and compare two arms across seeds.

Per-class accuracy = recall: of the cells truly of class *c*, the fraction predicted *c*. Computed
from each run's `test_pred.pt` (``argmax`` of the logits vs ``labels``), then averaged (mean ± sd) over
seeds. Prints a markdown table with support, per-arm accuracy, and Δ, plus balanced accuracy
(mean per-class recall). Class code order = the ``subclass`` categories in the dataset's train h5ad.

Metrics are recomputed from `test_pred.pt`, never read off a plot (memory `recompute-metrics-from-test-pred`).

This reproduces the "Per-class accuracy" table in `docs/20260820_deep_sample_shallow_loss.md`
(abundant top-6, 2-hop vs 4-hop, HalfHop-OFF, seeds 0-4). Edit `ARMS`/`DATA_SUBDIR` for other runs.

Usage (from repo root, gfs env):
    python -m gfs.analysis.per_class_accuracy
"""

import anndata as ad
import numpy as np
import torch

from gfs.utils import get_paths

# Dataset whose train h5ad defines the class-code -> subclass-name order.
DATA_SUBDIR = "0810_abudant_classes_only_top6"

# arm label -> {seed: log-dir basename under data/logs/}. Seeds must match across arms (paired).
ARMS = {
    "2-hop": {
        0: "20260820_182036_abundant_select5_seed0_nohh_s0",
        1: "20260823_160723_abundant_select5_seed1_nohh_s1",
        2: "20260823_160723_abundant_select5_seed2_nohh_s2",
        3: "20260823_160723_abundant_select5_seed3_nohh_s3",
        4: "20260823_160723_abundant_select5_seed4_nohh_s4",
    },
    "4-hop": {
        0: "20260820_182036_abundant_4hop_seed0_nohh_s0",
        1: "20260823_160727_abundant_4hop_seed1_nohh_s1",
        2: "20260823_160723_abundant_4hop_seed2_nohh_s2",
        3: "20260823_220552_abundant_4hop_seed3_nohh_s3",
        4: "20260823_220619_abundant_4hop_seed4_nohh_s4",
    },
}


def per_class_recall(log_dir, n_classes):
    """Return (recall[n_classes], support[n_classes]) for one run's test_pred.pt."""
    o = torch.load(f"{log_dir}/test_pred.pt", map_location="cpu", weights_only=False)
    pred = o["predictions"].argmax(1).numpy()
    lab = o["labels"].numpy()
    rec = np.full(n_classes, np.nan)
    support = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        m = lab == c
        support[c] = int(m.sum())
        if m.sum():
            rec[c] = (pred[m] == c).mean()
    return rec, support


def collect(runs, logs_root, n_classes):
    """Stack per-seed recall -> (mean[n_classes], sd[n_classes], support[n_classes])."""
    per_seed, support = [], None
    for seed in sorted(runs):
        rec, support = per_class_recall(f"{logs_root}{runs[seed]}", n_classes)
        per_seed.append(rec)
    arr = np.array(per_seed)
    return arr.mean(0), arr.std(0), support


def main():
    paths = get_paths()
    data_root = paths["data_root"]
    logs_root = f"{data_root}logs/"
    cats = list(
        ad.read_h5ad(f"{data_root}{DATA_SUBDIR}/isocortex_train_final.h5ad", backed="r").obs["subclass"].cat.categories
    )
    n_classes = len(cats)

    arm_names = list(ARMS)
    stats = {name: collect(ARMS[name], logs_root, n_classes) for name in arm_names}

    a, b = arm_names[0], arm_names[1]
    ma, sa, sup = stats[a]
    mb, sb, _ = stats[b]

    print(f"| class (subclass) | test support | {a} acc | {b} acc | delta ({b}-{a}) |")
    print("|---|---|---|---|---|")
    for c in range(n_classes):
        print(
            f"| {cats[c]} | {sup[c]} | {ma[c]:.4f} ± {sa[c]:.4f} | {mb[c]:.4f} ± {sb[c]:.4f} | {mb[c] - ma[c]:+.4f} |"
        )
    bal_a, bal_b = np.nanmean(ma), np.nanmean(mb)
    print(
        f"| **balanced acc** (mean per-class recall) | {int(sup.sum())} | "
        f"{bal_a:.4f} | {bal_b:.4f} | {bal_b - bal_a:+.4f} |"
    )


if __name__ == "__main__":
    main()
