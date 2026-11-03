# Feature ablation (ABUNDANT subset): does the learned selector beat a frozen PERSIST panel? (4-hop)

**Status:** DONE 2026-11-02. Abundant counterpart of `docs/feature_ablation.md` (full 22-label data).
With-selection reference reused from `docs/20260820_deep_sample_shallow_loss.md`. Abundant PERSIST panels
generated (2026-09-07). Frozen-panel array (job `27095776`, tasks 0–4) finished 2026-09-28; F1 recomputed
from `test_pred.pt` (n=31,563; matches logged `test_f1_overall` exactly) + paired t + cell-level and
seed-level paired bootstraps vs the selector arm, 2026-11-02. **Headline: on the abundant subset the
selector barely edges the frozen PERSIST panel (+0.006 weighted / +0.005 macro) — the gap is far smaller
than on full data (+0.025 weighted) and vanishes at the seed level.** See Results / Findings below.

## Question
On the **abundant 7-label subset** (the easier subclass regime where 4-hop *does* beat 2-hop — memory
`4hop-does-not-beat-2hop-full-data`), holding the 4-hop complete-field backbone fixed: how much of its
performance comes from **learned feature selection** vs from simply having a good 5-gene panel + the graph
backbone? Freeze a pre-chosen PERSIST panel, train the same 4-hop model with **no selector**, and compare
to the end-to-end selecting model. Same design as `feature_ablation.md`, moved to the abundant data.

## Design
Abundant subset (`data/0810_abudant_classes_only_top6/`, 500 genes, **7 subclasses**). Two arms, five runs
each (frozen arm: five abundant PERSIST panels at fixed `rand_seed=0`; with-selection arm: `rand_seed=0–4`).
Compared as distributions, not paired per seed (seed handling differs — see Caveat 2).

| Arm | trainer | config | selector | panel | HalfHop |
|---|---|---|---|---|---|
| **1. Frozen panel (no selection)** | `antelope_4hop_panel.py` | `antelope_abundant_4hop_panel.yaml` | `fs_method=none` | fixed abundant PERSIST select-5 (per seed) | off |
| **2. With selection** (reference) | `antelope_4hop.py` | `antelope_abundant_4hop.yaml` | `fs_method=persist`, `n_select=5` | learned end-to-end | off |

**Everything else identical** to the abundant 4-hop model: `n_hops=4`, `loss_hops=2`, `trainmode=2`
(complete-field loss), `local_layers=2`, `n_labels=7`, `batch_size=64`, `xyz_status=true`, **HalfHop off**
(train+eval), 100 epochs, lr=1e-3 constant, root-only val/test. The frozen arm differs from the selecting
arm in exactly one way: `fs_method="none"` (no `logits` parameter, no soft-mask annealing) with the AnnData
sliced to the 5-gene panel as input (`gene_ch=5`).

### With-selection reference (arm 2) — already complete, reused
From `docs/20260820_deep_sample_shallow_loss.md` (4-hop / `trainmode=2`, HalfHop OFF, `n_select=5`, `bs=64`,
seeds 0–4; jobs `25273003` seed 0 + `25336509` seeds 1–4; F1 recomputed from `test_pred.pt`, n=31,563):

| seed | selection weighted | selection macro |
|---|---|---|
| 0 | 0.7933 | 0.7824 |
| 1 | 0.7792 | 0.7698 |
| 2 | 0.7876 | 0.7794 |
| 3 | 0.7895 | 0.7818 |
| 4 | 0.7759 | 0.7684 |
| **mean ± sd** | **0.7851 ± 0.0073** | **0.7764 ± 0.0067** |

### The frozen panels (abundant PERSIST select-5) — GENERATED 2026-09-07
PERSIST panels generated **on the abundant data** (`~/persist/main.sh` with `data_subdir=0810_abudant_classes_only_top6`),
seeds 0–4, non-upsampled (1 row/cell, `train_rows=74634`), post-label-fix, `exposure=none`, same abundant
train/val masks as GFS → `~/persist/results/persist_panel_select5_seed{0-4}_abundant.csv`. Read directly from
the `selected_genes` column (0-indexed):

| seed | panel (gene-column indices) | gene symbols |
|---|---|---|
| 0 | 2, 167, 185, 205, 480 | Grik3, Cldn5, Ighm, Unc5d, Dscaml1 |
| 1 | 1, 2, 83, 416, 480 | Col5a1, Grik3, A830036E02Rik, Adgrf5, Dscaml1 |
| 2 | 2, 5, 167, 205, 480 | Grik3, Syt6, Cldn5, Unc5d, Dscaml1 |
| 3 | 2, 167, 185, 435, 480 | Grik3, Cldn5, Ighm, Slc30a3, Dscaml1 |
| 4 | 2, 38, 167, 435, 480 | Grik3, Rorb, Cldn5, Slc30a3, Dscaml1 |

Gene indices **2 (Grik3)** and **480 (Dscaml1)** are common to all 5 seeds; index **167 (Cldn5)** appears in
4/5. This is a more stable backbone than the full-data PERSIST panels (only index 2 shared there).

**Caveat 1 (mask exposure):** non-upsampled PERSIST draws ~65× fewer Gumbel masks/epoch than GFS's selector
(memory `persist-exposure-arm-confirms-mask-mechanism`); this is vanilla-regime PERSIST, a legitimate frozen
baseline, not the mask-matched variant.
**Caveat 2 (seed handling differs):** the frozen arm fixes `rand_seed=0` for all five runs (panel is the only
thing varying); the with-selection arm varies `rand_seed=0–4` (there the seed produces its five distinct
learned panels). Not seed-matched run-for-run — compare as distributions over the five runs.

## How to run
```bash
# 0) Generate abundant PERSIST panels (user sbatch; persist conda env, from ~/persist repo root):
cd /home/donghyun.kim/persist
for s in 0 1 2 3 4; do sbatch main.sh 5 $s "" 100 0810_abudant_classes_only_top6; done
# -> results/persist_panel_select5_seed{0-4}_abundant.csv ; read the selected_genes column.

# 1) Panels already pasted into the case block of main_abundant_4hop_panel_seeds.sh (done 2026-09-27).

# 2) Frozen-panel arm (user sbatch; gfs env):
cd /home/donghyun.kim/graphFeatureSelect/scripts
sbatch main_abundant_4hop_panel_seeds.sh   # array 0-4, task id -> panel index, rand_seed=0 all runs
```
Frozen-arm log dirs: `data/logs/<datetime>_abundant_4hop_panel_panel{0-4}_nohh_s0/`.

## Runs / Slurm IDs
| Arm | panel/seed | Slurm job | log dir | status |
|---|---|---|---|---|
| abundant PERSIST panels | seeds 0–4 | 25585849–25585853 | `~/persist/results/persist_panel_select5_seed{0-4}_abundant.csv` | ✅ done 2026-09-07 |
| frozen panel | panel 0 | 27095776_0 | `data/logs/20260927_111452_abundant_4hop_panel_panel0_nohh_s0` | ✅ done 2026-09-28 |
| frozen panel | panel 1 | 27095776_1 | `data/logs/20260927_111454_abundant_4hop_panel_panel1_nohh_s0` | ✅ done |
| frozen panel | panel 2 | 27095776_2 | `data/logs/20260927_111452_abundant_4hop_panel_panel2_nohh_s0` | ✅ done |
| frozen panel | panel 3 | 27095776_3 | `data/logs/20260927_111452_abundant_4hop_panel_panel3_nohh_s0` | ✅ done |
| frozen panel | panel 4 | 27095776_4 | `data/logs/20260927_111452_abundant_4hop_panel_panel4_nohh_s0` | ✅ done |
| with selection | seed 0–4 | 25273003 + 25336509 | see `docs/20260820_deep_sample_shallow_loss.md` | ✅ done (reused) |

## Results
_(recompute weighted + macro F1 from `test_pred.pt`, n=31,563 test cells; never quote off a plot)_

Frozen arm indexed by **panel index** (all `rand_seed=0`); with-selection arm by **seed** — not paired
row-for-row (Caveat 2), read each column as its own distribution.

Frozen weighted F1 recomputed from `test_pred.pt` matches logged `test_f1_overall` exactly (all 5 panels,
n=31,563, 7 classes).

| index | frozen weighted | frozen macro | selection weighted | selection macro |
|---|---|---|---|---|
| 0 | 0.7636 | 0.7454 | 0.7933 | 0.7824 |
| 1 | 0.7857 | 0.7770 | 0.7792 | 0.7698 |
| 2 | 0.7751 | 0.7671 | 0.7876 | 0.7794 |
| 3 | 0.7776 | 0.7752 | 0.7895 | 0.7818 |
| 4 | 0.7934 | 0.7901 | 0.7759 | 0.7684 |

Summary (mean ± sd over the 5 runs):

| Arm | weighted (mean ± sd) | macro (mean ± sd) |
|---|---|---|
| Frozen panel (no selection) | 0.7791 ± 0.0112 | 0.7710 ± 0.0165 |
| With selection | **0.7851 ± 0.0073** | **0.7764 ± 0.0067** |

### Significance
Both arms pair on the panel-generation seed i ("seed i's panel", PERSIST-selected vs GFS-selected), same
convention as the full-data `docs/feature_ablation.md`. Delta = selection − frozen.

**Primary test — cell-level paired bootstrap** (`gfs/analysis/paired_cell_bootstrap.py` machinery, B=10k,
one shared cell resample applied to all 5 index-paired runs, differenced within index, rng seed 0):

| Metric | delta_obs (sel − frozen) | 95% bootstrap CI | two-sided p |
|---|---|---|---|
| Weighted | +0.0060 | [+0.0028, +0.0092] | 0.0004 |
| Macro | +0.0054 | [+0.0018, +0.0089] | 0.0034 |

Both CIs sit entirely above 0 → selection beats the frozen panel *on this test set*, but the effect is
tiny (+0.006 weighted vs the full-data +0.0252). Scope: cell-level uncertainty within this one biological
sample (n=31,563, 7 classes), not across-sample generalization.

**Secondary test — seed-level paired t + bootstrap** (does it generalize across training seeds? paired t
primary + Cohen dz + sign + paired bootstrap B=100k on the 5 seed-level differences, rng seed 0):

| Metric | mean diff (sel − frozen) | paired t | p | Cohen dz | sign | bootstrap 95% CI | bootstrap p |
|---|---|---|---|---|---|---|---|
| Weighted | +0.0060 | 0.736 | 0.503 | 0.329 | 3/5 + | [−0.0087, +0.0193] | 0.375 |
| Macro | +0.0054 | 0.549 | 0.612 | 0.246 | 3/5 + | [−0.0104, +0.0232] | 0.578 |

At the seed level neither metric clears α=0.05 (both bootstrap CIs straddle 0, both t-tests p>0.5, small
dz, only 3/5 pairs positive) — the edge is indistinguishable from seed-to-seed noise at n=5. Note the
single best run in the whole experiment is **frozen panel 4** (0.7934 wtd / 0.7901 macro), which beats
every selection run; it carries the same `{2,480}` backbone plus Slc30a3+Rorb. Caveat 2 applies (frozen
arm fixes model-seed 0, selection varies 0–4, so pairs 1–4 carry extra model-init noise → conservative
seed-level test).

## Findings
**The abundant 7-label subset CLOSES the selector-vs-frozen-panel gap rather than widening it.** On full
data the selector beat the frozen PERSIST panel by +0.0252 weighted (cell-level p≈0); on the abundant
subset the same comparison shrinks to **+0.0060 weighted / +0.0054 macro** — ~4× smaller, cell-level
significant (both CIs > 0) but no longer distinguishable at the seed level (paired t p>0.5, 3/5 sign,
bootstrap CIs straddle 0). The learned selector and a good frozen PERSIST panel are **effectively tied**
on this regime.

This is consistent with the panels themselves: the abundant PERSIST panels share a more stable backbone
(indices 2/Grik3 and 480/Dscaml1 in all 5 seeds, 167/Cldn5 in 4/5 — vs only index 2 shared on full data),
so a frozen panel nearly matches the end-to-end-learned one. Where full-data frozen F1 suffered from a
weak panel 1 (variance, not ceiling), here the frozen arm has a *higher* ceiling than any selection run
(frozen panel 4 = 0.7934 wtd tops the table). The remaining frozen-arm disadvantage is still variance
(sd 0.0112 wtd vs selection's 0.0073), not a lower ceiling.

Relates to memory `frozen-panel-confirms-panel-quality` (the panel carries most of the signal; selection
adds little once the panel is good) and `4hop-does-not-beat-2hop-full-data` (abundant is the easy regime).
Takeaway: on the easy abundant regime, "learned selection" is not buying meaningful accuracy over a
well-chosen fixed panel — the backbone genes do the work.
