"""Exact unpaired permutation test of 4-hop vs 2-hop, seed-order-agnostic (5 v 5).

WHY THIS EXISTS (and how it differs from the paired t-test / cell bootstrap)
---------------------------------------------------------------------------
The primary reported test (``gfs.analysis.gaba_grouping_f1``) is a **paired** t-test that matches
2-hop seed_s to 4-hop seed_s. But the two arms are *different architectures*, so "seed 0 <-> seed 0"
is an essentially arbitrary matching of training-stochasticity draws. This module instead treats each
comparison as two **independent groups of 5** per-seed F1 values and asks whether the 4-hop group is
shifted from the 2-hop group, with **no reference to seed ordering**.

Test: **exact two-sided permutation test** on the difference of group means.
    - pool the 10 per-seed F1 values (5 2-hop + 5 4-hop);
    - T_obs = mean(4-hop) - mean(2-hop);
    - enumerate ALL C(10,5)=252 ways to relabel 5 of the 10 as the "4-hop" group;
    - for each, T = mean(group) - mean(rest);
    - p = (# permutations with |T| >= |T_obs|) / 252.
Exact (full enumeration), distribution-free, order-agnostic. The observed split and its mirror always
qualify, so the two-sided p floor is 2/252 = 0.0079 (much better than paired Wilcoxon's 0.0625 at n=5,
and low enough that a clean separation can clear alpha=0.05).

Also prints, per comparison, the across-seed Pearson correlation between the arms: high positive
correlation means the seed pairing captured real shared variance (paired test more powerful); near-zero
means unpairing loses ~nothing and is the more honest test.

Per-seed F1 recomputed from each run's ``test_pred.pt`` (never off a plot), reusing the log-dir registry
and loaders in ``gfs.analysis.paired_cell_bootstrap``. F1 on the full (un-resampled) test set.

Usage (from repo root, gfs env):
    python -m gfs.analysis.unpaired_permutation                 # all comparisons
    python -m gfs.analysis.unpaired_permutation glut nn gaba    # a subset by name
"""

import itertools
import sys

import numpy as np
from sklearn.metrics import f1_score

from gfs.analysis.paired_cell_bootstrap import (
    COMPARISONS,
    SEEDS,
    find_run,
    load_pred_labels,
)
from gfs.utils import get_paths


def exact_perm_test(a2, a4):
    """Exact two-sided permutation test on mean(a4) - mean(a2). Returns (t_obs, p, n_perm)."""
    a2 = np.asarray(a2, dtype=np.float64)
    a4 = np.asarray(a4, dtype=np.float64)
    pooled = np.concatenate([a2, a4])
    n = a4.size  # group size (5)
    t_obs = a4.mean() - a2.mean()
    idx = range(pooled.size)
    n_ge = 0
    n_perm = 0
    for combo in itertools.combinations(idx, n):
        mask = np.zeros(pooled.size, dtype=bool)
        mask[list(combo)] = True
        t = pooled[mask].mean() - pooled[~mask].mean()
        n_perm += 1
        if abs(t) >= abs(t_obs) - 1e-12:
            n_ge += 1
    return float(t_obs), n_ge / n_perm, n_perm


def per_seed_f1(runs, labels, average):
    """Per-seed F1 (full test set) for one arm's list of log dirs, aligned to `labels`."""
    out = []
    for r in runs:
        pred, lab = load_pred_labels(r)
        if not np.array_equal(labels, lab):
            raise RuntimeError("labels differ across runs — split not fixed, comparison invalid")
        out.append(f1_score(labels, pred, average=average, zero_division=0))
    return np.array(out)


def run_comparison(name, spec, logs_root):
    print(f"\n### {name} — {spec['label']} ###")
    runs = {arm: [find_run(logs_root, spec[arm], s) for s in SEEDS] for arm in ("2-hop", "4-hop")}
    missing = [f"{arm} seed{s}" for arm in ("2-hop", "4-hop") for s, r in zip(SEEDS, runs[arm]) if r is None]
    if missing:
        print(f"  -> incomplete, skipping (missing: {', '.join(missing)})")
        return None

    _, labels = load_pred_labels(runs["2-hop"][0])
    out = {}
    for average in ("weighted", "macro"):
        a2 = per_seed_f1(runs["2-hop"], labels, average)
        a4 = per_seed_f1(runs["4-hop"], labels, average)
        t_obs, p, n_perm = exact_perm_test(a2, a4)
        # across-seed correlation (diagnostic: did seed pairing capture shared variance?)
        corr = float(np.corrcoef(a2, a4)[0, 1]) if a2.std() > 0 and a4.std() > 0 else float("nan")
        out[average] = {"a2": a2, "a4": a4, "delta": t_obs, "perm_p": p, "n_perm": n_perm, "corr": corr}
        print(
            f"  [{average:>8}]  4-hop {a4.mean():.4f}  2-hop {a2.mean():.4f}  "
            f"Difference (4-hop - 2-hop) {t_obs:+.4f}  exact permutation p {p:.4f} "
            f"(n_perm={n_perm})  across-seed corr {corr:+.3f}"
        )
    return out


def main():
    names = sys.argv[1:] or list(COMPARISONS)
    unknown = [n for n in names if n not in COMPARISONS]
    if unknown:
        raise SystemExit(f"unknown comparison(s): {unknown}\navailable: {list(COMPARISONS)}")
    logs_root = f"{get_paths()['data_root']}logs/"
    print("Exact unpaired permutation test (5 v 5, seed-order-agnostic), full C(10,5)=252 enumeration")
    for name in names:
        run_comparison(name, COMPARISONS[name], logs_root)


if __name__ == "__main__":
    main()
