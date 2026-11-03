"""Cell-level paired bootstrap of 4-hop vs 2-hop, shared cell sample across all 5 matched seeds.

WHAT THIS IS (and how it differs from the older bootstraps in the docs)
----------------------------------------------------------------------
Each 4-hop-vs-2-hop comparison has **5 matched seeds** per arm, and all test cells come from the
**same biological sample** (one mouse-isocortex split; the val/test partition is FIXED across seeds —
seeds only randomize training stochasticity). There is no higher-level biological unit (donor / sample)
to resample, so we bootstrap **test cells**. Crucially, one bootstrap draw of cell indices is applied to
**all 10 runs at once** (2-hop and 4-hop, all 5 seeds), the metric is recomputed per seed, differenced
within seed, then averaged across seeds:

    for replicate b:
        idx = sample N_test cell indices WITH replacement          # shared by all 10 runs
        for seed s in 0..4:
            M_4hop[b, s] = f1(labels[idx], pred_4hop[s][idx])
            M_2hop[b, s] = f1(labels[idx], pred_2hop[s][idx])
            d[b, s]      = M_4hop[b, s] - M_2hop[b, s]              # paired within seed
        delta[b] = mean_s d[b, s]                                   # average across the 5 seeds

    delta_obs = mean_s (f1_4hop[s] - f1_2hop[s])                    # full, unbootstrapped test set
    95% CI    = [quantile(delta, .025), quantile(delta, .975)]
    p (2-sided)= min(1, 2 * min(mean(delta <= 0), mean(delta >= 0)))

This is preferred over (a) the seed-level paired bootstrap (only 5 clusters -> anti-conservative) and
(b) the single-seed cell bootstrap (ignores 4 of the 5 seeds). It keeps 4-hop/2-hop pairing, keeps the
same bootstrap cell sample across all seeds, does NOT retrain, and uses saved per-cell predictions only.

SCOPE CAVEAT (printed with every result):
    All cells are from one biological sample; bootstrap inference reflects cell-level uncertainty within
    this sample and does not establish across-sample biological generalization.

Metrics recomputed from each run's ``test_pred.pt`` (never off a plot; memory `recompute-metrics-from-test-pred`).
Weighted + macro F1. B = 10,000 replicates, fixed RNG seed.

Usage (from repo root, gfs env):
    python -m gfs.analysis.paired_cell_bootstrap                 # all comparisons
    python -m gfs.analysis.paired_cell_bootstrap glut nn gaba    # a subset by name
"""

import glob
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

from gfs.utils import get_paths

SEEDS = [0, 1, 2, 3, 4]
B = 10_000
RNG_SEED = 0

SCOPE_NOTE = (
    "All cells are from one biological sample; bootstrap inference reflects cell-level uncertainty "
    "within this sample and does not establish across-sample biological generalization."
)

# Each comparison: 2-hop and 4-hop log-dir glob templates (per seed `s`). `{s}` -> seed.
# Templates are chosen so each resolves to exactly ONE completed run (has test_pred.pt); find_run
# errors on ambiguity. All 5 seeds of a comparison share one fixed val/test split.
COMPARISONS = {
    # doc: 4hop_2hop_comparison.md  (full 22-label isocortex)
    "full22": {
        "label": "full 22-label isocortex",
        "2-hop": "*_full_2hop_seed{s}_nohh_s{s}",
        "4-hop": "*_full_4hop_seed{s}_nohh_s{s}",
    },
    # doc: 20260820_deep_sample_shallow_loss.md  (abundant top-6 + other = 7 labels)
    "abundant": {
        "label": "abundant top-6 (7 labels)",
        "2-hop": "*_abundant_select5_seed{s}_nohh_s{s}",
        "4-hop": "*_abundant_4hop_seed{s}_nohh_s{s}",
    },
    # doc: 20260919_abundant_next6.md  (ranks 7-12 + other = 7 labels)
    "next6": {
        "label": "abundant next-6 (7 labels)",
        "2-hop": "*_next6_seed{s}_nohh_s{s}",
        "4-hop": "*_next6_4hop_seed{s}_nohh_s{s}",
    },
    # doc: 20260919_diverse6_balanced.md  (2 Glut / 2 Gaba / 2 NN + other = 7 labels)
    "diverse6_bal": {
        "label": "diverse6 balanced (7 labels)",
        "2-hop": "*_diverse6_bal_seed{s}_nohh_s{s}",
        "4-hop": "*_diverse6_bal_4hop_seed{s}_nohh_s{s}",
    },
    # doc: 20260919_diverse6_maxdisp.md  (1 Glut / 2 Gaba / 3 NN + other = 7 labels)
    "diverse6_maxdisp": {
        "label": "diverse6 maxdisp (7 labels)",
        "2-hop": "*_diverse6_maxdisp_seed{s}_nohh_s{s}",
        "4-hop": "*_diverse6_maxdisp_4hop_seed{s}_nohh_s{s}",
    },
    # doc: 20260907_subclass_grouping_deep_sample.md  (coarse single-family keep-sets)
    "glut": {
        "label": "deep_sample Glut (10 labels)",
        "2-hop": "*_glut_select5_seed{s}_nohh_s{s}",
        "4-hop": "*_glut_4hop_seed{s}_nohh_s{s}",
    },
    "nn": {
        "label": "deep_sample NN (8 labels)",
        "2-hop": "*_nn_select5_seed{s}_nohh_s{s}",
        "4-hop": "*_nn_4hop_seed{s}_nohh_s{s}",
    },
    "gaba": {
        "label": "deep_sample Gaba (5 labels)",
        "2-hop": "*_gaba_select5_seed{s}_nohh_s{s}",
        "4-hop": "*_gaba_4hop_seed{s}_nohh_s{s}",
    },
    # doc: 20260914_subclass_grouping_finegrained.md  (fine Gaba keep-sets)
    "gaba_fine": {
        "label": "finegrained gaba_fine (9 labels)",
        "2-hop": "*_gaba_fine_seed{s}_nohh_s{s}",
        "4-hop": "*_gaba_fine_4hop_seed{s}_nohh_s{s}",
    },
    "gaba_cge": {
        "label": "finegrained gaba_cge (4 labels)",
        "2-hop": "*_gaba_cge_seed{s}_nohh_s{s}",
        "4-hop": "*_gaba_cge_4hop_seed{s}_nohh_s{s}",
    },
    "gaba_mge": {
        "label": "finegrained gaba_mge (6 labels)",
        "2-hop": "*_gaba_mge_seed{s}_nohh_s{s}",
        "4-hop": "*_gaba_mge_4hop_seed{s}_nohh_s{s}",
    },
}


def find_run(logs_root, template, seed):
    """Resolve exactly one completed log dir (has test_pred.pt) for a (template, seed), else None."""
    matches = sorted(glob.glob(f"{logs_root}{template.format(s=seed)}"))
    matches = [m for m in matches if glob.glob(f"{m}/test_pred.pt")]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous log dirs for {template.format(s=seed)}: {matches}")
    return matches[0]


def load_pred_labels(log_dir):
    """Return (pred argmax int array, labels int array) from a run's test_pred.pt."""
    o = torch.load(f"{log_dir}/test_pred.pt", map_location="cpu", weights_only=False)
    return o["predictions"].argmax(1).numpy().astype(np.int64), o["labels"].numpy().astype(np.int64)


def _fast_f1(conf, average):
    """Weighted / macro F1 from a KxK confusion matrix (rows=true, cols=pred).

    Matches sklearn.f1_score(..., zero_division=0) when every class appears in the sample (always true
    for large cell resamples): macro is the unweighted mean over all K classes, weighted is support-
    weighted. Cross-checked against sklearn on the full (un-resampled) test set at load time.
    """
    tp = np.diag(conf).astype(np.float64)
    support = conf.sum(1).astype(np.float64)  # n_k: true count per class
    pred_tot = conf.sum(0).astype(np.float64)  # p_k: predicted count per class
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    if average == "weighted":
        tot = support.sum()
        return float((support * f1).sum() / tot) if tot > 0 else 0.0
    return float(f1.mean())  # macro


def _codes(true, pred, K):
    """Per-cell combined confusion code true*K+pred, for O(N) bincount-based confusion."""
    return (true * K + pred).astype(np.int64)


def cell_paired_bootstrap(preds_2hop, preds_4hop, labels, average, n_boot=B, rng_seed=RNG_SEED):
    """Shared-cell paired bootstrap. preds_* are lists of 5 argmax arrays aligned to `labels`.

    Returns dict: f1_2hop (5,), f1_4hop (5,), mean_2hop, mean_4hop, delta_obs, ci_lo, ci_hi, p.
    """
    K = int(labels.max()) + 1
    N = labels.size

    # Observed (full test set) per-seed F1 via sklearn (headline numbers, doc-comparable).
    f1_2 = np.array([f1_score(labels, p, average=average, zero_division=0) for p in preds_2hop])
    f1_4 = np.array([f1_score(labels, p, average=average, zero_division=0) for p in preds_4hop])
    delta_obs = float((f1_4 - f1_2).mean())

    # Precompute confusion codes per run; validate fast-F1 == sklearn on the full sample.
    codes_2 = [_codes(labels, p, K) for p in preds_2hop]
    codes_4 = [_codes(labels, p, K) for p in preds_4hop]
    for c, ref in zip(codes_2 + codes_4, list(f1_2) + list(f1_4)):
        full = _fast_f1(np.bincount(c, minlength=K * K).reshape(K, K), average)
        assert abs(full - ref) < 1e-9, f"fast-F1 mismatch vs sklearn ({full} != {ref})"

    rng = np.random.default_rng(rng_seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)  # shared across all 10 runs
        d = 0.0
        for c2, c4 in zip(codes_2, codes_4):
            conf2 = np.bincount(c2[idx], minlength=K * K).reshape(K, K)
            conf4 = np.bincount(c4[idx], minlength=K * K).reshape(K, K)
            d += _fast_f1(conf4, average) - _fast_f1(conf2, average)
        deltas[b] = d / len(codes_2)

    ci_lo, ci_hi = np.quantile(deltas, [0.025, 0.975])
    p = min(1.0, 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return {
        "f1_2hop": f1_2,
        "f1_4hop": f1_4,
        "mean_2hop": float(f1_2.mean()),
        "mean_4hop": float(f1_4.mean()),
        "delta_obs": delta_obs,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p": float(p),
    }


def run_comparison(name, spec, logs_root):
    print(f"\n### {name} — {spec['label']} ###")
    runs = {arm: [find_run(logs_root, spec[arm], s) for s in SEEDS] for arm in ("2-hop", "4-hop")}
    missing = [f"{arm} seed{s}" for arm in ("2-hop", "4-hop") for s, r in zip(SEEDS, runs[arm]) if r is None]
    if missing:
        print(f"  -> incomplete, skipping (missing: {', '.join(missing)})")
        return None

    preds_2, labels = [], None
    for r in runs["2-hop"]:
        p, lab = load_pred_labels(r)
        preds_2.append(p)
        if labels is None:
            labels = lab
        elif not np.array_equal(labels, lab):
            raise RuntimeError(f"{name}: labels differ across 2-hop seeds — split not fixed, pairing invalid")
    preds_4 = []
    for r in runs["4-hop"]:
        p, lab = load_pred_labels(r)
        if not np.array_equal(labels, lab):
            raise RuntimeError(f"{name}: labels differ between arms — pairing by cell index invalid")
        preds_4.append(p)

    print(f"  n_test_cells={labels.size}  n_classes={int(labels.max()) + 1}  seeds={SEEDS}  B={B}")
    out = {}
    for average in ("weighted", "macro"):
        res = cell_paired_bootstrap(preds_2, preds_4, labels, average)
        out[average] = res
        print(
            f"  [{average:>8}]  4-hop: {res['mean_4hop']:.4f}  2-hop: {res['mean_2hop']:.4f}  "
            f"Difference (4-hop - 2-hop): {res['delta_obs']:+.4f}  "
            f"95% bootstrap CI: [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
            f"Two-sided bootstrap p-value: {res['p']:.4f}"
        )
    return out


def main():
    names = sys.argv[1:] or list(COMPARISONS)
    unknown = [n for n in names if n not in COMPARISONS]
    if unknown:
        raise SystemExit(f"unknown comparison(s): {unknown}\navailable: {list(COMPARISONS)}")
    logs_root = f"{get_paths()['data_root']}logs/"
    print(f"Cell-level paired bootstrap (shared cell sample across 5 seeds), B={B}, rng_seed={RNG_SEED}")
    print(SCOPE_NOTE)
    for name in names:
        run_comparison(name, COMPARISONS[name], logs_root)
    print(f"\n{SCOPE_NOTE}")


if __name__ == "__main__":
    main()
