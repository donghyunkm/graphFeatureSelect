"""Cell-level paired bootstrap for the feature-ablation comparison (selection vs frozen PERSIST panel).

Companion to the seed-level paired t + bootstrap in ``docs/feature_ablation.md``. Same machinery as
``gfs/analysis/paired_cell_bootstrap.py`` (shared cell resample across all 5 index-paired runs, metric
recomputed per index, differenced within index, averaged), but for the two feature-ablation arms:

    arm A ("frozen"):    frozen PERSIST select-5 panel, no selector (model-seed 0, panel index i)
    arm B ("selection"): learned end-to-end 4-hop selector (model-seed i)

paired on the panel-generation seed i (frozen index i = PERSIST-seed-i panel; selection index i =
GFS-seed-i panel), both on the same full data / val-test split. Delta = selection - frozen (sel - frozen).

CAVEATS (carried from the doc):
- The frozen arm fixes the model-training seed at 0 while the selection arm uses seed i, so pairs 1-4
  carry extra model-init noise inside the difference -> inflates within-pair variance, conservative.
- SCOPE: all cells are one biological sample; this reflects cell-level uncertainty within the sample,
  NOT across-sample biological generalization. It measures within-sample direction/power, not
  generalization -- the seed-level bootstrap in the doc remains the honest n=5 generalization estimate.

Usage (from repo root, gfs env):
    python -m gfs.analysis.feature_ablation_cell_bootstrap
"""

import numpy as np

from gfs.analysis.paired_cell_bootstrap import (
    RNG_SEED,
    SCOPE_NOTE,
    SEEDS,
    B,
    cell_paired_bootstrap,
    find_run,
    load_pred_labels,
)
from gfs.utils import get_paths

# `{s}` -> panel-generation seed / index. Frozen arm fixes model-seed 0 (`_nohh_s0`); selection varies.
FROZEN_TEMPLATE = "*_full_4hop_panel_panel{s}_nohh_s0"
SELECTION_TEMPLATE = "*_full_4hop_seed{s}_nohh_s{s}"


def main():
    logs_root = f"{get_paths()['data_root']}logs/"
    print(
        f"Cell-level paired bootstrap: selection vs frozen PERSIST panel (feature ablation), B={B}, rng_seed={RNG_SEED}"
    )
    print(SCOPE_NOTE)
    print("\n### feature_ablation — full 22-label isocortex, 4-hop backbone (delta = selection - frozen) ###")

    frozen_runs = [find_run(logs_root, FROZEN_TEMPLATE, s) for s in SEEDS]
    sel_runs = [find_run(logs_root, SELECTION_TEMPLATE, s) for s in SEEDS]
    missing = [f"frozen idx{s}" for s, r in zip(SEEDS, frozen_runs) if r is None] + [
        f"selection seed{s}" for s, r in zip(SEEDS, sel_runs) if r is None
    ]
    if missing:
        raise SystemExit(f"incomplete (missing: {', '.join(missing)})")

    # Load frozen arm; anchor the shared label vector / val-test split.
    preds_frozen, labels = [], None
    for r in frozen_runs:
        p, lab = load_pred_labels(r)
        preds_frozen.append(p)
        if labels is None:
            labels = lab
        elif not np.array_equal(labels, lab):
            raise RuntimeError("labels differ across frozen panels — split not fixed, pairing invalid")
    preds_sel = []
    for r in sel_runs:
        p, lab = load_pred_labels(r)
        if not np.array_equal(labels, lab):
            raise RuntimeError("labels differ between arms — pairing by cell index invalid")
        preds_sel.append(p)

    print(f"  n_test_cells={labels.size}  n_classes={int(labels.max()) + 1}  paired indices={SEEDS}  B={B}")
    # cell_paired_bootstrap computes delta = arm_B - arm_A; pass frozen as A, selection as B.
    for average in ("weighted", "macro"):
        res = cell_paired_bootstrap(preds_frozen, preds_sel, labels, average)
        print(
            f"  [{average:>8}]  selection: {res['mean_4hop']:.4f}  frozen: {res['mean_2hop']:.4f}  "
            f"Difference (selection - frozen): {res['delta_obs']:+.4f}  "
            f"95% bootstrap CI: [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
            f"Two-sided bootstrap p-value: {res['p']:.4f}"
        )
    print(f"\n{SCOPE_NOTE}")


if __name__ == "__main__":
    main()
