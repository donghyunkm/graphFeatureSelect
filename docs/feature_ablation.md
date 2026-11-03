# Feature ablation: does the learned selector beat a frozen PERSIST panel? (4-hop, full data)

**Status:** ✅ **COMPLETE** (2026-09-02). Both arms finished all 5 runs. Frozen-panel arm (`25511099`,
all 5 panels) F1 recomputed from `test_pred.pt` (matches logged `test_f1_overall` exactly).

**Answer: the learned selector beats a frozen PERSIST panel on this test set — significantly on both
weighted and macro F1 by a cell-level paired bootstrap.** Frozen panel = **0.6847 ± 0.0254 weighted /
0.5198 ± 0.0084 macro** vs with-selection **0.7099 ± 0.0142 / 0.5272 ± 0.0141** (+0.0252 weighted /
+0.0074 macro for selection). The **cell-level paired bootstrap** (n=31562 test cells, B=10k) puts both
gaps entirely above 0 — weighted CI [+0.0229, +0.0274], macro CI [+0.0040, +0.0108], both p≈0 — so
selection beats frozen on this sample, decisively on weighted, by a small-but-significant margin on macro.
The margin is largely **variance-avoidance**: the frozen arm is much noisier (weighted sd 0.025 vs 0.014),
pulled down by panel 1 (PERSIST `{2,25,83,207,225}`, 0.6430); the other four frozen panels land 0.68–0.71
and overlap the selector. So a good frozen PERSIST panel on the same 4-hop backbone gets most of the way,
and the selector's edge is "it never picks a bad panel" more than "every learned panel is much better."
(Across-*seed* generalization is a separate, weaker claim — at n=5 the seed-level paired t is not
significant; see the Significance subsection.)

## Question
Holding the 4-hop complete-field backbone fixed, how much of its performance comes from the **learned
feature selection** vs from simply having a good 5-gene panel + the graph backbone? I.e. if we **freeze**
a pre-chosen panel and train the same 4-hop model with **no selector**, do we match the end-to-end
selecting model?

## Design
Full data (500 genes, 22 subclasses). Two arms, five runs each (frozen arm: five PERSIST panels at fixed
`rand_seed=0`; with-selection arm: `rand_seed=0–4`). Compared as distributions, not paired per seed —
see Caveat 2 below.

| Arm | trainer | config | selector | panel |
|---|---|---|---|---|
| **1. Frozen panel (no selection)** | `antelope_4hop_panel.py` | `antelope_4hop_panel.yaml` | `fs_method=none` | fixed PERSIST select-5 panel (per seed) |
| **2. With selection** | `antelope_4hop.py` | `antelope_4hop.yaml` | `fs_method=persist`, `n_select=5` | learned end-to-end |

**Everything else is identical** to the 4-hop model in `docs/4hop_2hop_comparison.md`: `n_hops=4`,
`loss_hops=2`, `trainmode=2` (complete-field loss), `local_layers=2`, `n_labels=22`, `batch_size=64`,
`xyz_status=true`, **HalfHop off** (train+eval), 100 epochs, lr=1e-3 constant, root-only val/test. The
frozen arm differs from the selecting arm in exactly one way: `fs_method="none"` (no `logits`
parameter, no soft-mask annealing) with the AnnData sliced to the 5-gene panel as the model input
(`gene_ch=5`).

### The frozen panels (PERSIST select-5, 0-indexed)
Each frozen run freezes the PERSIST panel from the correspondingly-seeded PERSIST run (panel index 0–4),
but the **model training seed is fixed to `rand_seed=0` for all five frozen runs** — since each run
already differs by its panel, fixing the seed removes init/shuffle noise as a confound so the only thing
varying across the five is the panel itself. (The with-selection arm still varies its seed 0–4, because
there the seed *is* what produces its five distinct learned panels; fixing it would collapse those runs.)

> **Provenance (corrected 2026-08-30).** The panels are being **regenerated** by the upstream PERSIST
> library (`github.com/iancovert/persist`, pristine — verified no local edits to the package) via
> `/home/donghyun.kim/persist/main.py`, run on the **same full data and train/val masks** as GFS
> (`isocortex_{train,valtest}_final.h5ad`, 22 subclasses, 500 genes), **non-upsampled** (1 row/cell),
> **post label-fix**. Jobs `25507771`–`25507775` (seeds 0–4) → `persist/results/persist_panel_select5_seed{0-4}.csv`.
>
> This *replaces* the panels previously taken from `persist_genes_selected_genes.csv`, which came from an
> **external, pre-label-fix PERSIST run** (label-encoding bug, `docs/old-claude.md:944`; predates the
> 2026-07-29 fix) and are therefore compromised — do not use them.
>
> Caveat 1 (mask exposure): non-upsampled PERSIST draws ~65× fewer Gumbel masks/epoch than GFS's
> selector (memory `persist-exposure-arm-confirms-mask-mechanism`); this is vanilla-regime PERSIST, a
> legitimate frozen baseline, not the mask-matched variant.
>
> Caveat 2 (seed handling differs across arms): the frozen arm fixes the model training seed to
> `rand_seed=0` for all five runs (the panel is the only thing that varies), whereas the with-selection
> arm varies `rand_seed=0–4` (there the seed is what produces its five distinct learned panels).
> **Update (2026-09-27):** the arms *can* still be paired on the **panel-generation seed** — frozen index
> i is PERSIST-seed-i's panel, selection index i is GFS-seed-i's panel, both on the same data/masks — so
> index i vs index i is a legitimate seed-for-seed contrast of the *selection method* (PERSIST vs GFS).
> The residual asymmetry is only that the frozen arm fixes the *model-training* seed at 0 while the
> selection arm uses seed i, so pairs 1–4 carry extra model-init noise inside the difference; this
> inflates within-pair variance and makes the paired test **conservative**, not anti-conservative. The
> paired t + bootstrap in the Significance subsection above use this pairing. (Reading the arms as two
> independent 5-run distributions and running an unpaired/Welch test is also valid and reaches the same
> ns conclusion.)

**Generated 2026-08-31** — jobs `25507771`–`25507775` all completed cleanly (each ended `done, selected
5 genes` → wrote its CSV; PERSIST stops on target-count convergence, so ~66–85 epochs, not the 100 cap).
CSVs at `/home/donghyun.kim/persist/results/persist_panel_select5_seed{0-4}.csv` (confirm `exposure=none`,
`train_rows=74633` non-upsampled). Panels below are read directly from those CSVs' `selected_genes`.

| seed | panel (gene-column indices) | gene symbols |
|---|---|---|
| 0 | 2, 12, 174, 435, 480 | Grik3, Ptprm, Tmem215, Slc30a3, Dscaml1 |
| 1 | 2, 25, 83, 207, 225 | Grik3, Slc17a6, A830036E02Rik, Zfp536, Nxph1 |
| 2 | 2, 83, 250, 375, 480 | Grik3, A830036E02Rik, Clic5, Kcnmb2, Dscaml1 |
| 3 | 2, 12, 25, 115, 435 | Grik3, Ptprm, Slc17a6, Sorcs3, Slc30a3 |
| 4 | 2, 83, 86, 271, 435 | Grik3, A830036E02Rik, Sulf2, Tgfbr2, Slc30a3 |

Only gene index **2** (Grik3) is common to all 5 seeds; the rest vary substantially seed-to-seed — much
less stable than the GFS selector's near-fixed {2, 12, 83, 480} panel (see `docs/4hop_2hop_comparison.md`).

## How to run
```bash
cd /home/donghyun.kim/graphFeatureSelect/scripts
sbatch main_fulldata_4hop_panel_seeds.sh   # array 0-4, frozen-panel arm (arm 1)
# arm 2 already submitted: main_fulldata_4hop_seeds.sh (Slurm 25507593)
```
Frozen-arm log dirs: `data/logs/<datetime>_full_4hop_panel_seed{0-4}_nohh_s{0-4}/`.

## Runs / Slurm IDs
Frozen arm submitted 2026-08-31 as **25511099** (`--array=0-4`; task id → panel index, all at
`rand_seed=0`). Log dirs: `data/logs/<datetime>_full_4hop_panel_panel{0-4}_nohh_s0/`.

| Arm | panel/seed | Slurm job | log dir | status |
|---|---|---|---|---|
| frozen panel | panel 0 | 25511099_0 | `20260831_063916_full_4hop_panel_panel0_nohh_s0` | ✅ done (100 ep) |
| frozen panel | panel 1 | 25511099_1 | `20260831_223946_full_4hop_panel_panel1_nohh_s0` | ✅ done (100 ep) |
| frozen panel | panel 2 | 25511099_2 | `20260901_045330_full_4hop_panel_panel2_nohh_s0` | ✅ done (100 ep) |
| frozen panel | panel 3 | 25511099_3 | `20260901_050109_full_4hop_panel_panel3_nohh_s0` | ✅ done (100 ep) |
| frozen panel | panel 4 | 25511099_4 | `20260901_054330_full_4hop_panel_panel4_nohh_s0` | ✅ done (100 ep) |
| with selection | seed 0–4 | 25507593_{0-4} | (see `docs/4hop_2hop_comparison.md`) | ✅ done (5 seeds) |

## Results
_(recompute weighted + macro F1 from `test_pred.pt`; never quote off a plot)_

Frozen arm is indexed by **panel index** (all at `rand_seed=0`); with-selection arm by **seed** — the two
are not paired row-for-row (Caveat 2), so read each column as its own distribution. Selection column from
`docs/4hop_2hop_comparison.md` (job 25507593).

Frozen weighted F1 recomputed from `test_pred.pt` matches logged `test_f1_overall` exactly (verified
panels 0 & 1). Frozen arm indexed by **panel index** (all `rand_seed=0`); selection arm by **seed**.

| index | frozen weighted | frozen macro | selection weighted | selection macro |
|---|---|---|---|---|
| 0 | 0.7111 | 0.5302 | 0.6849 | 0.5056 |
| 1 | 0.6430 | 0.5184 | 0.7185 | 0.5318 |
| 2 | 0.6907 | 0.5095 | 0.7145 | 0.5246 |
| 3 | 0.6835 | 0.5262 | 0.7190 | 0.5297 |
| 4 | 0.6950 | 0.5146 | 0.7125 | 0.5442 |

Summary (mean ± sd over the 5 runs):

| Arm | weighted (mean ± sd) | macro (mean ± sd) |
|---|---|---|
| Frozen panel (no selection) | 0.6847 ± 0.0254 | 0.5198 ± 0.0084 |
| With selection | **0.7099 ± 0.0142** | **0.5272 ± 0.0141** |

### Significance
Both arms pair on the panel-generation seed i ("seed i's panel", PERSIST-selected vs GFS-selected) — see
revised Caveat 2 for why this pairing is defensible.

**Primary test — cell-level paired bootstrap (does selection beat frozen on this test set?).**
The question of interest is whether the selector beats the frozen panel *on this data*, so the primary
test resamples test **cells**. Same machinery as 4hop-vs-2hop (`MEMORY.md`
`significance-convention-ttest-primary`; `gfs/analysis/feature_ablation_cell_bootstrap.py`, B=10k, one
shared cell resample applied to all 5 index-paired runs, differenced within index, delta = selection −
frozen), run 2026-09-27:

| Metric | delta_obs (sel − frozen) | 95% bootstrap CI | two-sided p |
|---|---|---|---|
| Weighted | +0.0252 | [+0.0229, +0.0274] | 0.0000 |
| Macro | +0.0074 | [+0.0040, +0.0108] | 0.0000 |

**Selection significantly beats the frozen PERSIST panel on both metrics** — both CIs sit entirely above 0
(never crosses in 10k resamples). Weighted (+0.025) is the solid win; macro (+0.007) is small but
significant. Scope: all test cells are one biological sample (n=31562, 22 classes), so this is cell-level
uncertainty *within this sample*, not across-sample biological generalization.

**Secondary test — seed-level paired t (does it generalize across training seeds?).** Convention:
paired t primary + Cohen dz + sign + paired bootstrap (B=100k on the 5 seed-level differences, `rng` seed 0).

| Metric | mean diff (sel − frozen) | paired t | p | Cohen dz | sign | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| Weighted | +0.0252 | 1.544 | 0.198 | 0.690 | 4/5 + | [−0.0051, +0.0548] |
| Macro | +0.0074 | 0.820 | 0.458 | 0.367 | 4/5 + | [−0.0094, +0.0211] |

At the seed level neither metric clears α=0.05 (both bootstrap CIs straddle 0) — the edge is not
distinguishable from seed-to-seed noise at n=5. Weighted shows a medium-large effect (dz=0.69) but is
dominated by seed 1 (+0.0755, the panel lacking the `{12,480}` backbone) and largely evaporates without
it; the one negative pair is **seed 0** (−0.0262), the only fully model-seed-matched pair (pairs 1–4 carry
extra model-init noise because the frozen arm fixes model-seed 0 while selection uses seed i — this
inflates within-pair variance and makes the seed-level test conservative). This is the weaker,
across-seed-generalization claim; it does not contradict the cell-level result, which is the one that
answers "does selection beat frozen here."

## Findings
- **The learned selector significantly beats a frozen PERSIST panel on this test set** on the same 4-hop
  backbone: +0.0252 weighted / +0.0074 macro, both p≈0 by the cell-level paired bootstrap (weighted CI
  [+0.0229, +0.0274], macro CI [+0.0040, +0.0108]; n=31562 cells). Weighted is the solid win; macro is
  small-but-significant. (Across training seeds the gap is *not* significant at n=5 — weighted p=0.198,
  macro p=0.458 by the seed-level paired t — but that is the weaker generalization claim; see Significance.)
- **The frozen arm's disadvantage is variance, not a uniformly lower ceiling.** Four of five PERSIST
  panels score 0.68–0.71 weighted (panel 0 = 0.7111 actually *exceeds* the selector's seed-0 0.6849);
  only panel 1 (`{2,25,83,207,225}`, the panel that lacks the `{2,12,83,480}` backbone — no 12, no 480)
  drops to 0.6430 and drags the mean down. The selector's win is largely "it never picks a bad panel,"
  not "its panels are individually much better."
- **So most of the 4-hop performance is "having a good 5-gene panel + the graph backbone," not the
  end-to-end selection itself** — consistent with selection being graph-invariant and with the prior
  frozen-panel finding (MEMORY.md `frozen-panel-confirms-panel-quality`: at a matched pipeline the
  selector's panel beat PERSIST's, but both framings were amended). The gene-index-2 (Grik3) + `{12,83,480}`
  backbone is what carries the score; panels missing it (panel 1) underperform.
- Caveat: not paired seed-for-seed (Caveat 2) — read each column as its own 5-run distribution.
