# Subclass-grouping variants: deep-sample (4-hop) vs 2-hop across Glut / NN / Gaba

**Cell-type families:** **Glut** = glutamatergic (excitatory) neurons, **NN** = **non-neuronal** cells
(glia + vascular: astrocytes, oligodendrocytes, endothelial, etc.), **Gaba** = GABAergic (inhibitory) neurons.

**Date:** 2026-09-07 · **Updated:** 2026-09-14 · **Status:** **COMPLETE** — array `25585854` finished
(all 30 runs, 3 datasets × 2 arms × 5 seeds). See the **cross-family verdict** at the bottom.
As of 2026-09-14: **all three datasets done (5/5 both arms each).** Gaba 4-hop finished (last task
`25585854_29`, gaba 4-hop seed4, 2026-09-13 23:49). See the Gaba significance/panels blocks below.
As of 2026-09-13: **Glut and NN (non-neuronal) both arms fully done (5/5 each)**; Gaba 4-hop seeds 3-4 still
training. See the Glut and NN significance blocks below.
As of 2026-09-11: **Glut both arms fully done (5/5 each)** — see the Glut significance block below.
As of 2026-09-10: **Glut both arms done** (4hop missing seed4), **NN 2-hop done**, **NN 4-hop running**
(seed0 restarted — see below), **Gaba both arms pending** (`QOSGrpMemLimit`/`JobArrayTaskLimit`). Results
below are recomputed from `test_pred.pt` (n=31,562), never off a plot (memory `recompute-metrics-from-test-pred`).

## Question
Does the 4-hop complete-field training win (`docs/20260820_deep_sample_shallow_loss.md`, +0.0132 w / +0.0181 m
on the top-6 abundant subset) hold up under **different subclass groupings**? We rerun the same
2-hop/tm0-vs-4-hop/tm2 comparison on three new "abundant-style" datasets that keep a different family of
subclasses and collapse the rest into `other`. This probes whether the complete-field benefit is specific to
the original top-6 label set or generalizes across cortical cell families.

## Datasets (built 2026-09-07)
All three are **relabelings of the full 22-label data** — keep a chosen subclass family, collapse everything
else (including the full data's pre-existing `other`) into a single `other`. All cells kept; `X`, graph
(`obsp["spatial_connectivities"]`, 1,492,660 edges), spatial coords, and the **fine-stratified full-data
`val_mask`/`test_mask`** copied verbatim. Built via `gfs/data/build_subclass_variant.py`.

| dataset | dir | labels | kept subclasses | train `other` |
|---|---|---|---|---|
| **Glut** | `20260907_glut_classes` | 10 | 9 Glut layer types (001,004,005,006,007,022,029,030,032) | 31,107 |
| **NN**   | `20260907_nn_classes`   | 8  | 7 non-neuronal (319,326,327,330,331,333,334) | 52,328 |
| **Gaba** | `20260907_gaba_classes` | 5  | 4 GABAergic (046,049,052,053) | 68,845 |

**Split provenance (verified):**
- Same cells as full data — 74,633 train / 63,124 valtest — masks copied → **no leakage**, train↔valtest
  boundary identical across all three (and to full/abundant).
- **Stratified 50/50 val:test**: every class incl. collapsed `other` is within **1 cell** of exact 50/50.
  This is inherited from the full-data split, which is stratified at the fine 22-subclass level (so any
  collapse of fine classes stays balanced) — NOT re-drawn per dataset.
- **`020 L2/3 IT RSP Glut` dropped** from the Glut keep-set (folded into `other`): it has 791 train cells but
  **0 valtest cells** (the valtest file is missing subclass 020 — documented full-data quirk), so keeping it
  would create an 11th class that is trained but never evaluated. NN/Gaba unaffected (020 is the only
  missing-from-valtest subclass). Glut therefore has 9 kept + other = 10 labels.

> Caveat: these are **not** on the same test cells as the original abundant deep_sample run (that build
> re-randomized its val/test split — ~52% disagreement with full data). They can't be compared head-to-head
> anyway (different label sets); each variant is internally consistent (its 2-hop and 4-hop arms share one split).

## Design
Per dataset, two arms × five seeds (0–4), matching the deep_sample HalfHop-OFF protocol exactly:

| Arm | trainer | config | n_hops | trainmode | HalfHop | n_select |
|---|---|---|---|---|---|---|
| **2-hop baseline** | `antelope.py` | `antelope_{glut,nn,gaba}.yaml` | 2 | 0 | off | 5 |
| **4-hop complete-field** | `antelope_4hop.py` | `antelope_{glut,nn,gaba}_4hop.yaml` | 4 (loss_hops=2) | 2 | off | 5 |

`n_select=5`, `halfhop=false`, `batch_size=64`, `local_layers=2`, `xyz_status=true`, 100 epochs, lr=1e-3
constant, best-`val_f1_overall` checkpoint for test — all baked into the dedicated configs. Only `rand_seed`
(0–4) varies per run (array task id → seed).

## How to run
```bash
cd /home/donghyun.kim/graphFeatureSelect/scripts

# Option A (recommended): all 30 runs in ONE array, throttled to 4 concurrent (rest queue):
sbatch main_subclass_grouping_all.sh   # --array=0-29%4; task id -> (dataset, arm, seed)

# Option B: per-arm arrays (array 0-4 = seed). NOTE: %4 on each caps 4 *per arm*, not globally —
# submitting all 6 could run up to 24 at once. Use Option A for a true 4-job global cap.
sbatch main_glut_2hop_seeds.sh
sbatch main_nn_2hop_seeds.sh
sbatch main_gaba_2hop_seeds.sh
sbatch main_glut_4hop_seeds.sh
sbatch main_nn_4hop_seeds.sh
sbatch main_gaba_4hop_seeds.sh
```
Log dirs: `data/logs/<datetime>_{tag}_select5_seed{0-4}_nohh_s{0-4}/` (2-hop);
`data/logs/<datetime>_{tag}_4hop_seed{0-4}_nohh_s{0-4}/` (4-hop).

## Runs / Slurm IDs
Launched 2026-09-07 as **one throttled array `25585854`** (`main_subclass_grouping_all.sh`, `--array=0-29%4`,
≤4 concurrent). Task id → (dataset, arm, seed): dataset = id/10 (0=glut,1=nn,2=gaba), arm = (id%10)/5
(0=2-hop,1=4-hop), seed = id%5.

| dataset | arm | array task ids | status (2026-09-10) |
|---|---|---|---|
| Glut | 2-hop | 25585854_[0-4] | **DONE** (5/5 `test_pred.pt`) |
| Glut | 4-hop | 25585854_[5-9] | **DONE (5/5)**; seed4 completed as requeued job `25601706_9`, dir `20260909_173319_glut_4hop_seed4_nohh_s4` (the earlier `20260909_104057` seed4 dir is an aborted partial — ignore) |
| NN | 2-hop | 25585854_[10-14] | **DONE** (5/5) |
| NN | 4-hop | 25585854_[15-19] | **DONE (5/5)**; seed0 = restart dir `20260910_222140` (task 15 cancelled+restarted as `25629969`; stale `20260909_173231` @ep50 ignored); seed4 dir `20260910_193926` |
| Gaba | 2-hop | 25585854_[20-24] | **DONE (5/5)** |
| Gaba | 4-hop | 25585854_[25-29] | **DONE (5/5)** (seed3 dir `20260912_144741`, seed4 dir `20260912_185931`; finished 2026-09-13 23:49) |

**seed0 NN-4hop restart (2026-09-10):** the original task `25585854_15` (dir `20260909_173231_nn_4hop_seed0_nohh_s0`)
ran ~2× slower than its siblings and its `val_f1_overall` oscillated wildly through the anneal (min 0.065,
max 0.748) instead of converging like seeds 1–3 (~0.84–0.88) — an unstable/bad-GPU run, not a crash. Cancelled
(`scancel 25585854_15`) and relaunched via `scripts/main_nn_4hop_seed0_restart.sh` (`--array=0`, identical
config) → job **`25629969_0`**, currently **PENDING on `QOSGrpMemLimit`** (group 128 GB quota saturated; will
start when a long-running job frees memory). The old collapsed seed0 dir is stale — ignore it for the final set.

## Results _(recomputed from `test_pred.pt`, n=31,562; population sd, ddof=0; 2026-09-10)_

Per-seed weighted / macro F1 (verified):

| dataset | arm | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| Glut | 2-hop (select5) | 0.8354 / 0.7181 | 0.8355 / 0.7333 | 0.8411 / 0.7540 | 0.8450 / 0.7400 | 0.8409 / 0.7390 |
| Glut | 4-hop | 0.8477 / 0.7310 | 0.8521 / 0.7636 | 0.8460 / 0.7208 | 0.8518 / 0.7376 | 0.8492 / 0.7624 |
| NN | 2-hop (select5) | 0.8691 / 0.5101 | 0.8751 / 0.5609 | 0.8779 / 0.5658 | 0.8715 / 0.4835 | 0.8719 / 0.4734 |
| NN | 4-hop | 0.8751 / 0.5634 | 0.8905 / 0.5568 | 0.8771 / 0.5665 | 0.8956 / 0.5772 | 0.8706 / 0.5082 |
| Gaba | 2-hop (select5) | 0.9841 / 0.8900 | 0.9830 / 0.8880 | 0.9848 / 0.8900 | 0.9828 / 0.8689 | 0.9825 / 0.8838 |
| Gaba | 4-hop | 0.9845 / 0.8900 | 0.9848 / 0.8920 | 0.9843 / 0.8962 | 0.9846 / 0.8876 | 0.9840 / 0.8998 |

Per-dataset summary (mean ± sd):

| dataset | 2-hop wtd | 2-hop macro | 4-hop wtd | 4-hop macro | Δ wtd | Δ macro |
|---|---|---|---|---|---|---|
| Glut | 0.8396 ± 0.0037 (n=5) | 0.7369 ± 0.0116 | 0.8494 ± 0.0023 (n=5) | 0.7431 ± 0.0171 | **+0.0098** | +0.0062 |
| NN (non-neuronal) | 0.8731 ± 0.0031 (n=5) | 0.5187 ± 0.0384 | 0.8818 ± 0.0096 (n=5) | 0.5544 ± 0.0240 | +0.0087 (ns, p=0.15) | +0.0357 (ns, p=0.12) |
| Gaba | 0.9834 ± 0.0009 (n=5) | 0.8841 ± 0.0080 | 0.9844 ± 0.0003 (n=5) | 0.8931 ± 0.0043 | +0.0010 (ns, p=0.088) | +0.0090 (ns, p=0.066) |

## Glut: full 5-seed comparison + significance _(2026-09-11, both arms n=5)_
Paired by seed — the val/test split is **identical across all seeds** (masks copied verbatim; `rand_seed`
only reseeds init/training), so 2-hop seed_i and 4-hop seed_i score the exact same 31,562 test cells. Paired
tests are the right family; Wilcoxon reported alongside because n=5 is small.

| metric | Δ (4hop − 2hop) mean ± sd | per-seed Δ | paired t | Wilcoxon | bootstrap p | Cohen dz |
|---|---|---|---|---|---|---|
| **weighted F1** | **+0.0098 ± 0.0047** | [+.0123, +.0166, +.0049, +.0068, +.0083] | t=4.64, **p=0.0097** | W=0, p=0.0625 | **<0.0002** (CI [+.0078,+.0118]) | **2.07** |
| macro F1 | +0.0062 ± 0.0253 | [+.0129, +.0303, −.0332, −.0024, +.0234] | t=0.55, p=0.61 | W=6, p=0.81 | 0.0104 (CI [+.0015,+.0108]) | 0.25 |

_(Bootstrap redone 2026-09-25 = **cell-level paired bootstrap**, B=10 000 (`gfs/analysis/paired_cell_bootstrap.py`): a shared test-cell resample is applied across all 5 seeds × both arms, per-seed F1 differenced within seed then averaged; 95% percentile CI + 2-sided p. ⚠️ Scope: this is **cell-level uncertainty within one biological sample — NOT across-sample/donor generalization**; it is far higher-powered than the seed-level paired t-test and can disagree with it. The paired t-test stays the primary/reported test. Supersedes the earlier seed-level bootstrap (B=50 000). Here the cell-level bootstrap now flags **macro significant (p=0.0104)** too, whereas the t-test calls macro a tie (p=0.61) — so the robust, t-confirmed result is the **weighted win** (t p=0.0097); the macro "significance" is a within-sample power artifact, not a cross-seed effect.)_

- **Weighted F1: 4-hop wins, and it's real** — improvement in **all 5/5 seeds**, paired t p≈0.01, very large
  effect (dz≈2.1). Wilcoxon p=0.0625 is the *smallest attainable* p for n=5 (every sign positive), i.e. the
  rank test is maximally in favor and only fails α=0.05 for lack of samples — consistent with the t-test.
- **Macro F1: genuine tie** (p=0.61, dz=0.25); per-seed sign flips both ways. Deep sampling does not help the
  rare Glut layer types on average. So the weighted win is driven by the abundant classes / collapsed `other`.
- Direction matches the abundant top-6 deep-sample win (`20260820_deep_sample_shallow_loss.md`); magnitude
  (+0.0098 w) is smaller than that run's +0.0132 w.

### Glut panels (final-epoch hard selection, n=5 each; index → freq)
| arm | gene frequencies | modal panel |
|---|---|---|
| 2-hop | `12`×5, `83`×5, `2`×4, `334`×4, `106`×3, `480`×2, `435`×1, `267`×1 | no single mode; `{2,12,83}` core + `334` |
| 4-hop | `2`×5, `12`×5, `83`×5, `5`×3, `480`×3, `106`×2, `435`×2 | **`{2,5,12,83,480}`** (3/5) |

- Shared backbone **`{12,83}`** in both arms (5/5); 4-hop also fixes `2` at 5/5 (2-hop 4/5).
- **The signature shift: 4-hop drops gene `334` entirely** (4/5 in 2-hop → **0/5** in 4-hop) and promotes
  **gene `5`** (0/5 → 3/5) plus `480` (2/5 → 3/5). So the deeper receptive field consistently swaps ~2 genes:
  it stops picking `334` and prefers the `{5,480}` pair. This is the same "depth changes which genes look
  informative" pattern noted on abundant/full — deep sampling reshapes the panel, not just the score.

## NN (non-neuronal): DONE (5/5 both arms) _(2026-09-13)_
All seeds complete. NN 4-hop seed0 = restart dir `20260910_222140` (the stale `20260909_173231` @ ep50 is
ignored); seed4 = `20260910_193926`. F1 recomputed from `test_pred.pt` (n=31,562), population sd.

| arm | mean wtd | mean macro | seeds 0-4 wtd | seeds 0-4 macro |
|---|---|---|---|---|
| 2-hop | 0.8731 ± 0.0031 | 0.5187 ± 0.0384 | .8691/.8751/.8779/.8715/.8719 | .5101/.5609/.5658/.4835/.4734 |
| 4-hop | 0.8818 ± 0.0096 | 0.5544 ± 0.0240 | .8751/.8905/.8771/.8956/.8706 | .5634/.5568/.5665/.5772/.5082 |

| metric | Δ (4−2) | per-seed Δ | paired t (n=5) | Wilcoxon | bootstrap p | Cohen dz |
|---|---|---|---|---|---|---|
| weighted | +0.0087 | [+.0060, +.0154, −.0008, +.0241, −.0013] | t=1.78, p=0.15 (ns) | W=3, p=0.31 | <0.0002 (CI [+.0074,+.0100]) | 0.79 |
| macro | +0.0357 | [+.0533, −.0041, +.0007, +.0937, +.0348] | t=1.98, p=0.12 (ns) | W=2, p=0.19 | <0.0002 (CI [+.0303,+.0411]) | 0.89 |

_(Bootstrap redone 2026-09-25 = **cell-level paired bootstrap**, B=10 000 (`gfs/analysis/paired_cell_bootstrap.py`): shared test-cell resample across all 5 seeds × both arms, per-seed F1 differenced within seed then averaged; 95% percentile CI + 2-sided p. Both metrics now come out p<0.0002. ⚠️ Scope: this is **cell-level uncertainty within one biological sample — NOT across-sample/donor generalization**; it is far higher-powered than the seed-level paired t-test and only corroborates *direction*. On the primary paired t-test **NN stays ns** (weighted p=0.15, macro p=0.12) — per-seed diffs flip sign twice, so the cross-seed effect is not established. The t-test is the reported call. Supersedes the earlier seed-level bootstrap (B=50 000).)_

- **NN: 4-hop trends up on both metrics but NEITHER is significant at n=5.** Weighted Δ=+0.0087 (p=0.15),
  macro Δ=+0.0357 (p=0.12). Effect sizes are medium-large (dz≈0.8–0.9) but per-seed differences **flip sign
  twice** (seeds 2 & 4 slightly favor 2-hop), so the mean gain is not robust. Contrast with Glut, where
  weighted improved in **all 5/5** seeds (p=0.0097). The macro trend rests heavily on seeds 0 & 3.
- **This is the headline NN result and it does not replicate the clean Glut win** — consistent with
  `4hop-does-not-beat-2hop-full-data` (deep sampling's benefit is not universal across label sets).

### NN panels (final-epoch hard selection, n=5 each; index → freq)
| arm | gene frequencies | backbone |
|---|---|---|
| 2-hop | `45`×5, `137`×5, `461`×4, `271`×3, `28`×3, `325`×2, `495`×2, `189`×1 | `{45,137}` core, `461` strong |
| 4-hop | `137`×5, `45`×4, `271`×2, `325`×2, `461`×2, `178`×2, `495`×2, `123`×2, `167,28,44,416`×1 | `{137,45}` only |

Index → symbol: `45`=Slc7a10 (astro), `137`=Sox10 (oligo), `461`=Chrm3, `271`=Tgfbr2, `28`=Kcnk9,
`325`=Hmcn1, `495`=Prom1, `189`=Osbpl3, `178`=Sv2b, `123`=Cntn6, `167`=Cldn5 (endo), `44`=Aqp4 (astro),
`416`=Adgrf5.

- **Shared core in both arms: `Sox10`(137) + `Slc7a10`(45)** — the load-bearing NN markers (oligo + astro).
- **2-hop is tight, 4-hop is diffuse:** 2-hop locks `{45,137,461}` and a stable `{271,28}` support set; 4-hop
  keeps `137` (5/5) and mostly `45` (4/5) but **scatters the other three slots** (`178,123` ×2; `167,44,416`
  singletons — vascular/finer markers). Same "depth reshapes the panel" pattern as Glut, but here it produces
  a **less reproducible** panel rather than a clean marker swap.
- **Per-seed 4-hop-vs-2-hop overlap:** 4/5, 3/5, 4/5, **1/5**, 4/5 — seed3's 4-hop panel diverges completely
  (`{44,123,137,178,416}`) yet scores the *highest* 4-hop F1 (0.8956 wtd / 0.5772 macro): multiple
  near-equivalent panels exist for NN.

## Gaba: DONE (5/5 both arms) + significance _(2026-09-14)_
All seeds complete (Gaba 4-hop finished 2026-09-13 23:49, last array task `25585854_29`). Paired by seed
(identical val/test split across seeds). F1 recomputed from `test_pred.pt` (n=31,562), population sd.

| arm | mean wtd | mean macro | seeds 0-4 wtd | seeds 0-4 macro |
|---|---|---|---|---|
| 2-hop | 0.9834 ± 0.0009 | 0.8841 ± 0.0080 | .9841/.9830/.9848/.9828/.9825 | .8900/.8880/.8900/.8689/.8838 |
| 4-hop | 0.9844 ± 0.0003 | 0.8931 ± 0.0043 | .9845/.9848/.9843/.9846/.9840 | .8900/.8920/.8962/.8876/.8998 |

| metric | Δ (4−2) | per-seed Δ | paired t (n=5) | Wilcoxon | bootstrap p | Cohen dz |
|---|---|---|---|---|---|---|
| weighted | +0.0010 | [+.0004, +.0018, −.0005, +.0018, +.0015] | t=2.25, p=0.088 (ns) | W=2, p=0.19 | 0.0036 (CI [+.0003,+.0017]) | 1.01 |
| macro | +0.0090 | [+.0001, +.0040, +.0062, +.0187, +.0159] | t=2.51, p=0.066 (ns) | W=0, p=0.0625 | 0.0024 (CI [+.0033,+.0146]) | 1.12 |

_(Bootstrap redone 2026-09-25 = **cell-level paired bootstrap**, B=10 000 (`gfs/analysis/paired_cell_bootstrap.py`): shared test-cell resample across all 5 seeds × both arms, per-seed F1 differenced within seed then averaged; 95% percentile CI + 2-sided p (weighted 0.0036, macro 0.0024). ⚠️ Scope: this is **cell-level uncertainty within one biological sample — NOT across-sample/donor generalization**; it is far higher-powered than the seed-level paired t-test and corroborates *direction* only. On the primary paired t-test **Gaba stays ns** (0.088 / 0.066); weighted is near ceiling (0.984) with almost no headroom. The t-test is the reported call → **ns**. Supersedes the earlier seed-level bootstrap (B=50 000).)_

- **Gaba: 4-hop trends up on BOTH metrics, neither crosses α=0.05 at n=5, but the direction is very clean.**
  Weighted wins 4/5 (Δ=+0.0010, p=0.088) — but Gaba weighted F1 is near ceiling (0.984), so there is almost no
  headroom; the tiny Δ has large-but-noisy effect size (dz≈1.0).
- **Macro is the more informative axis here: 4-hop wins 5/5 seeds** (Δ=+0.0090, dz≈1.1). Wilcoxon p=0.0625 is
  the *smallest attainable* p for n=5 (all signs positive) — maximally in favor, failing α=0.05 only for
  sample size, exactly like Glut's weighted result. The paired t (p=0.066) agrees. So on the rare/minority
  Gaba classes (where the macro average lives), the deeper receptive field gives a small but perfectly
  consistent lift.
- Contrast the axis with Glut: for Glut the clean 5/5 win was on **weighted** and macro tied; for Gaba it is
  the reverse (macro 5/5, weighted 4/5 & near-ceiling). Both point the same way (4-hop ≥ 2-hop everywhere),
  but which metric shows it depends on where each dataset has headroom.

### Gaba panels (final-epoch hard selection, n=5 each; index → freq)
| arm | gene frequencies | backbone |
|---|---|---|
| 2-hop | `130`×5, `156`×4, `225`×4, `298`×4, `140`×2, `425`×2, `487`×1, `63,389,479`×1 | `{130,156,225,298}` |
| 4-hop | `487`×5, `130`×4, `156`×4, `298`×3, `63`×2, `225`×2, `265,28,479,389,427`×1 | `{130,156}` + `487` locked |

Index → symbol: `130`=Elfn1, `156`=**Gad2** (canonical GABA marker), `225`=Nxph1, `298`=Oprd1,
`487`=**Synpr** (synaptoporin, interneuron), `63`=**Slc32a1** (VGAT, pan-GABAergic), `140`=Kit,
`425`=Npas1, `389`=Reln, `479`=Grpr, `265`=Bmp3, `28`=Kcnk9, `427`=Vip.

- **Shared backbone in both arms: `{Elfn1(130), Gad2(156)}`** — the load-bearing pan-GABAergic pair
  (`Gad2` is the canonical marker), stable at ≥4/5 in each arm.
- **The signature shift: 4-hop promotes `Synpr`(487) from 1/5 → 5/5** (now its single most-stable gene) and
  demotes `Nxph1`(225) from 4/5 → 2/5. So the deeper field consistently swaps in `Synpr` (an interneuron
  synaptic-vesicle marker) for `Nxph1`. `Oprd1`(298) stays strong in both (4/5 → 3/5).
- **Per-seed 4-hop-vs-2-hop overlap:** 3/5, 3/5, 3/5, 4/5, 3/5 — a steady ~2-gene swap (mostly the
  `225→487` trade plus one support slot), more reproducible than NN's scattered 4-hop panel and comparable to
  Glut's clean `334→{5,480}` swap. Same "depth reshapes the panel, not just the score" pattern all three
  datasets show.

## Cross-family verdict _(2026-09-14, all three datasets 5/5 both arms)_
| dataset | Δ weighted (p) | Δ macro (p) | 4-hop clean-win axis | panel shift (2-hop → 4-hop) |
|---|---|---|---|---|
| **Glut** | **+0.0098 (p=0.0097, 5/5)** ✅ | +0.0062 (p=0.61, tie) | weighted | drops `334`, adds `{5,480}` |
| **NN** | +0.0087 (p=0.15) | +0.0357 (p=0.12) | none (signs flip) | `{Sox10,Slc7a10}` core, 4-hop scatters 3 slots |
| **Gaba** | +0.0010 (p=0.088, 4/5, ceiling) | +0.0090 (p=0.066, **5/5**) | macro | `{Gad2,Elfn1}` core, swaps `Nxph1→Synpr` |

- **Direction is unanimous: all 6 dataset×metric mean Δ are positive** — 4-hop never loses on the mean in any
  family, on either metric. But **only Glut-weighted reaches significance** (p=0.01); NN never does (per-seed
  signs flip), and Gaba's cleanest signal (macro, 5/5 seeds) sits exactly at the n=5 Wilcoxon floor (p=0.0625).
- **The 4-hop benefit is real in direction but small and rarely significant at n=5** — consistent with
  `4hop-does-not-beat-2hop-full-data` (the deep-sample win does not universally clear significance across
  label sets). Where a dataset has weighted headroom (Glut) the win shows on weighted; where weighted is at
  ceiling (Gaba, 0.984) it shows on macro (rare classes); NN is genuinely ambiguous.
- **Panels: every family shows "depth reshapes the panel"** — 4-hop keeps each family's canonical marker
  backbone (Glut `{12,83}`; NN `Sox10`+`Slc7a10`; Gaba `Gad2`+`Elfn1`) but consistently swaps ~2 of the
  remaining slots. For Glut/Gaba this is a clean, reproducible marker swap; for NN it degrades panel stability.
- **Practical takeaway:** 4-hop complete-field training is a safe, mildly beneficial default (never worse on
  the mean, sometimes clearly better), but on these ~5-gene panels the gain is not large enough to be reliably
  significant with 5 seeds. To claim it for NN/Gaba would need more seeds; Glut-weighted already stands.

## Findings _(all datasets complete — 2026-09-14; earlier partial notes retained below)_
- **Glut: DONE (5/5 both arms).** 4-hop wins weighted F1 (+0.0098, p=0.0097, 5/5 seeds), macro tied — see the
  Glut significance block above.
- **NN (non-neuronal): DONE (5/5 both arms).** 4-hop trends up (wtd +0.0087 p=0.15, macro +0.0357 p=0.12) but
  **neither is significant at n=5** — per-seed differences flip sign twice, unlike Glut's clean 5/5 weighted
  win. NN 4-hop panel is notably less stable than 2-hop's (backbone `Sox10`+`Slc7a10`; 4-hop scatters the
  other 3 slots). See NN block.
- **Gaba: DONE (5/5 both arms).** 2-hop wtd 0.9834 ± 0.0009 / macro 0.8841 ± 0.0080 (very high; easiest of the
  three groupings). 4-hop wtd 0.9844 ± 0.0003 / macro 0.8931 ± 0.0043 — trends up on both, neither significant
  at n=5 (wtd p=0.088 near ceiling; **macro wins 5/5, p=0.066, Wilcoxon at the n=5 floor**). Panel swaps
  `Nxph1`→`Synpr`; backbone `{Gad2,Elfn1}`. See the Gaba significance/panels blocks above.
- **NN (non-neuronal) 2-hop:** high weighted (0.873) but **low macro (0.52 ± 0.038)** — the collapsed `other`
  + a few abundant non-neuronal classes dominate; rare NN classes are largely missed. 4-hop lifts macro to
  0.554 but not significantly (see NN block).
- **Panels (final-epoch hard selection, gene index → freq across completed seeds):**
  - Glut: see the n=5 Glut panels table above (4-hop drops `334`, promotes `{5,480}`).
  - NN 2-hop: backbone `{45,137}` (5/5), `461` (4/5); + `271`(3/5),`28`(3/5). Panels e.g. `{45,137,271,325,461}`.
  - (col `sel_*` in `selections.csv`; col0 is `epoch`, NOT a gene — don't include it. `prob_*` ≈ 0.99 at ep99 = converged hard selection.)
- **Nothing open — all three families complete.** See the **Cross-family verdict** block above: direction is
  unanimous (all 6 dataset×metric Δ positive) but only Glut-weighted clears α=0.05; NN never does, Gaba's
  macro sits at the n=5 Wilcoxon floor. Echoes `4hop-does-not-beat-2hop-full-data` — the deep-sample win is
  real in direction but small and rarely significant across label sets at n=5.
