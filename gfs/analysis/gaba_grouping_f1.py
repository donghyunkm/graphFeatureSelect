"""Recompute weighted + macro F1 (2-hop vs 4-hop) for the 09-14 Gaba fine-grained grouping runs.

For each dataset (gaba_fine / gaba_cge / gaba_mge) this:
  1. auto-discovers the per-seed log dirs for both arms (2-hop baseline, 4-hop complete-field),
  2. recomputes weighted-F1 and macro-F1 per seed from `test_pred.pt` (``argmax`` of logits vs labels),
     never off a plot (memory `recompute-metrics-from-test-pred`),
  3. reports mean ± population sd (ddof=0) per arm and Δ (4-hop minus 2-hop),
  4. runs paired-by-seed significance: paired t-test, Wilcoxon signed-rank, Cohen's dz,
  5. prints the final selected gene panel (last row of `selections.csv` -> gene_symbol) per arm/seed
     and the cross-seed selection frequency.

Arms/seeds are paired (same seed = same val/test split). Datasets or arms without a complete set of
`test_pred.pt` are skipped with a note, so this can be run mid-array and re-run once the rest land.

Reference — 09-07 coarse Gaba (4 canonical classes, ceiling-bound): 2-hop -> 4-hop
weighted 0.9834 -> 0.9844, macro 0.8841 -> 0.8931. See `docs/20260907_subclass_grouping_deep_sample.md`.

Usage (from repo root, gfs env):
    python -m gfs.analysis.gaba_grouping_f1
"""

import glob

import anndata as ad
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import f1_score

from gfs.utils import get_paths

SEEDS = [0, 1, 2, 3, 4]

# dataset key -> (data subdir defining subclass order, human label)
DATASETS = {
    "fine": ("20260914_gaba_fine", "gaba_fine (9 lbl)"),
    "cge": ("20260914_gaba_cge", "gaba_cge (4 lbl)"),
    "mge": ("20260914_gaba_mge", "gaba_mge (6 lbl)"),
}

# arm label -> log-dir glob template (per dataset key `ds` and seed `s`)
ARMS = {
    "2-hop": "*_gaba_{ds}_seed{s}_nohh_s{s}",
    "4-hop": "*_gaba_{ds}_4hop_seed{s}_nohh_s{s}",
}

REF_COARSE_GABA = "09-07 coarse Gaba (ref): weighted 0.9834->0.9844, macro 0.8841->0.8931"


def find_run(logs_root, ds, seed, template):
    """Resolve exactly one log dir for (dataset, seed, arm), or None if absent."""
    matches = sorted(glob.glob(f"{logs_root}{template.format(ds=ds, s=seed)}"))
    matches = [m for m in matches if glob.glob(f"{m}/test_pred.pt")]  # require completed run
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous log dirs for {ds} seed{seed}: {matches}")
    return matches[0]


def f1s(log_dir):
    """Return (weighted_f1, macro_f1) for one run's test_pred.pt."""
    o = torch.load(f"{log_dir}/test_pred.pt", map_location="cpu", weights_only=False)
    pred = o["predictions"].argmax(1).numpy()
    lab = o["labels"].numpy()
    return (
        f1_score(lab, pred, average="weighted", zero_division=0),
        f1_score(lab, pred, average="macro", zero_division=0),
    )


def final_panel(log_dir, symbols):
    """Final selected gene symbols (last epoch of selections.csv)."""
    import csv

    with open(f"{log_dir}/selections.csv") as fh:
        last = list(csv.DictReader(fh))[-1]
    idx = [int(last[k]) for k in last if k.startswith("sel_")]
    return idx, [symbols[i] for i in idx]


def paired_stats(a2, a4):
    """Paired significance of (4-hop - 2-hop) across seeds. Returns dict."""
    d = np.array(a4) - np.array(a2)
    out = {"mean_delta": d.mean(), "t_p": np.nan, "wilcoxon_p": np.nan, "cohen_dz": np.nan}
    if len(d) >= 2 and d.std(ddof=1) > 0:
        out["t_p"] = stats.ttest_rel(a4, a2).pvalue
        out["cohen_dz"] = d.mean() / d.std(ddof=1)
        try:
            out["wilcoxon_p"] = stats.wilcoxon(a4, a2).pvalue
        except ValueError:
            pass
    return out


def analyse_dataset(ds_key, subdir, label, logs_root, data_root):
    symbols = list(ad.read_h5ad(f"{data_root}{subdir}/isocortex_train_final.h5ad", backed="r").var["gene_symbol"])

    # gather per-seed runs, keeping only seeds present in BOTH arms (paired)
    runs = {arm: {} for arm in ARMS}
    for arm, tmpl in ARMS.items():
        for s in SEEDS:
            r = find_run(logs_root, ds_key, s, tmpl)
            if r:
                runs[arm][s] = r
    paired_seeds = sorted(set(runs["2-hop"]) & set(runs["4-hop"]))

    print(f"\n### {label} ###")
    for arm in ARMS:
        have = sorted(runs[arm])
        print(f"  {arm}: {len(have)}/5 complete (seeds {have})")
    if not paired_seeds:
        print("  -> no seed complete in both arms yet; skipping metrics.")
        return
    print(f"  paired seeds used: {paired_seeds}")

    metrics = {arm: {"w": [], "m": []} for arm in ARMS}
    panels = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for s in paired_seeds:
            w, m = f1s(runs[arm][s])
            metrics[arm]["w"].append(w)
            metrics[arm]["m"].append(m)
            _, syms = final_panel(runs[arm][s], symbols)
            panels[arm].append((s, syms))

    for metric, name in [("w", "weighted-F1"), ("m", "macro-F1")]:
        a2, a4 = metrics["2-hop"][metric], metrics["4-hop"][metric]
        st = paired_stats(a2, a4)
        print(
            f"  {name}: 2-hop {np.mean(a2):.4f}±{np.std(a2):.4f} | "
            f"4-hop {np.mean(a4):.4f}±{np.std(a4):.4f} | "
            f"Δ {st['mean_delta']:+.4f} | t p={st['t_p']:.4f} | "
            f"Wilcoxon p={st['wilcoxon_p']:.4f} | Cohen dz={st['cohen_dz']:+.3f}"
        )

    for arm in ARMS:
        freq = {}
        for _, syms in panels[arm]:
            for g in syms:
                freq[g] = freq.get(g, 0) + 1
        top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        print(f"  {arm} panel selection freq (x/{len(paired_seeds)} seeds): " + ", ".join(f"{g}:{c}" for g, c in top))


def main():
    paths = get_paths()
    data_root = paths["data_root"]
    logs_root = f"{data_root}logs/"
    print(REF_COARSE_GABA)
    for ds_key, (subdir, label) in DATASETS.items():
        analyse_dataset(ds_key, subdir, label, logs_root, data_root)


if __name__ == "__main__":
    main()
