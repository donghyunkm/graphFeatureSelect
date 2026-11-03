# progress.md

_Living tracker. Update at the end of meaningful work; convert relative dates to absolute._
_Concise chronological log — detailed write-ups live in `docs/` (see `docs/README.md`); research history in `MEMORY.md`._

## Session log — 2026-09-27

**Feature ablation (FULL DATA) — significance added to `docs/feature_ablation.md`.** The doc previously
had only mean±sd and a "~1 combined sd" hand-wave; no formal test. Added a paired t + Cohen dz + sign +
paired bootstrap (B=100k on the 5 differences), pairing frozen-index-i vs selection-index-i on the
**panel-generation seed** (frozen index i = PERSIST-seed-i's panel; selection index i = GFS-seed-i's
learned panel). Revised Caveat 2 accordingly: this pairing IS defensible; the only residual asymmetry is
the frozen arm fixes model-training seed=0 while selection uses seed=i, which inflates within-pair
variance (makes the test **conservative**, not anti-conservative). Result: **neither metric significant**
— weighted +0.0252 (t=1.544, p=0.198, dz=0.69, 4/5+, CI [−0.0051,+0.0548]); macro +0.0074 (t=0.820,
p=0.458, dz=0.37, 4/5+, CI [−0.0094,+0.0211]). Weighted effect is medium-large but dominated by seed 1
(+0.0755, the panel lacking the {12,480} backbone); the one negative pair is seed 0 (the only fully
model-seed-matched pair). Computed ad-hoc via scipy (numbers taken from the doc's per-run table); no new
compute, no committed analysis script.

**Update (later 2026-09-27): added the CELL-LEVEL paired bootstrap as the PRIMARY test and reframed the
doc's headline around it.** User's focus is "does selection beat frozen on this test set," not seed-level
generalization. New committed script `gfs/analysis/feature_ablation_cell_bootstrap.py` (reuses
`paired_cell_bootstrap.py` machinery: B=10k, one shared test-cell resample across all 5 index-paired runs,
differenced within index, delta = selection − frozen). Result: **selection significantly beats frozen on
both metrics** — weighted +0.0252 CI [+0.0229,+0.0274] p≈0; macro +0.0074 CI [+0.0040,+0.0108] p≈0
(n=31562 cells, 22 classes). delta_obs matches the doc's per-run means exactly. Scope: within-sample
(one biological sample), not across-sample generalization. `docs/feature_ablation.md` Answer + Findings +
Significance rewritten to lead with this; the ns seed-level paired t is now the SECONDARY
(across-seed-generalization) claim, not the headline. No retraining; predictions loaded from `test_pred.pt`.

**Feature ablation (ABUNDANT 7-label subset) — frozen-panel arm LAUNCHED.** `docs/feature_ablation_abundant.md`
was scaffolded 2026-09-07 with the abundant PERSIST panels generated (jobs 25585849–853) but the frozen
arm never run. This session: read the 5 abundant PERSIST panels from
`~/persist/results/persist_panel_select5_seed{0-4}_abundant.csv` (verified `exposure=none`,
`train_rows=74634` non-upsampled), pasted them into `scripts/main_abundant_4hop_panel_seeds.sh` (replacing
the full-data placeholder + "DO NOT LAUNCH" TODOs), filled the panel table in the doc, and the **user
launched the frozen array as job `27095776`** (tasks 0–4, `rand_seed=0` all, HalfHop off). Panels:
s0 `[2,167,185,205,480]`, s1 `[1,2,83,416,480]`, s2 `[2,5,167,205,480]`, s3 `[2,167,185,435,480]`,
s4 `[2,38,167,435,480]` — backbone {2 Grik3, 480 Dscaml1} in all 5, {167 Cldn5} in 4/5 (more stable than
full-data PERSIST). The **selector (with-selection) arm is already complete** and reused: 5 nohh seeds
(`data/logs/20260820_182036_abundant_4hop_seed0_nohh_s0` + `..._seed{1-4}_nohh_s{1-4}`), all with
`test_pred.pt`; F1 0.7851±0.0073 weighted / 0.7764±0.0067 macro (from `docs/20260820_deep_sample_shallow_loss.md`).
**PENDING:** when `27095776` finishes → recompute F1 from each `test_pred.pt`, fill Results table, run the
same paired t + bootstrap vs the selector arm, write Findings (compare to full-data: ns there).

## Session log — 2026-11-02

**Feature ablation (ABUNDANT) — DONE.** Job `27095776` (tasks 0–4) finished 2026-09-28 (all hit
`max_epochs=100`, clean test). Recomputed weighted+macro F1 from each `test_pred.pt` (n=31,563, 7 classes;
weighted matches logged `test_f1_overall` exactly). **Frozen panel 0.7791±0.0112 wtd / 0.7710±0.0165 macro
vs selection 0.7851±0.0073 / 0.7764±0.0067** → delta +0.0060 wtd / +0.0054 macro. Cell-level paired
bootstrap (primary): both CIs > 0 (wtd [+0.0028,+0.0092] p=0.0004; macro [+0.0018,+0.0089] p=0.0034) but
tiny. Seed-level paired t + bootstrap (secondary): ns (t<0.74, p>0.5, dz<0.33, 3/5 sign, bootstrap CIs
straddle 0). **Headline: on the easy abundant regime the selector only ~ties a good frozen PERSIST panel —
gap ~4× smaller than full-data's +0.0252; backbone genes {2 Grik3, 480 Dscaml1} do the work.**
`docs/feature_ablation_abundant.md` Results/Significance/Findings filled, Runs table marked done.
**Queue now clear** (user to confirm `squeue`); `outs/` has nothing newer than the 2026-09-28 `27095776` tasks.

**`docs/conclusion.md` written + wired into READMEs (top-level results summary).** New `docs/conclusion.md`
pulls the most relevant experiments into one story with two claims: (1) classification accuracy — the GNN on
its real edges wins every comparison (selector+proper > selector+shuffled = +0.082 `does-the-graph-help.md`;
selector+proper vs PERSIST+no-graph = 0.7103 vs 0.5724 default-PERSIST, +0.138; selector+proper vs
PERSIST+proper = +0.0252 full / +0.0060 abundant, `feature_ablation*.md`); (2) gene panel — the 4-hop
selector's panel is best (2-hop shuffle picks the same panel = sample-efficiency not spatial; 4-hop picks a
different, better panel — `deep_sample_shallow_loss.md`, `diverse6_balanced.md`). Caveat recorded in the doc:
the panel in claims 1.1/1.2 (`[2,12,83,435,480]`) is the **2-hop** selector panel; redo with the 4-hop panel
is a PREDICTION, not verified (1.3 already uses 4-hop selector genes). Linked from top-level `README.md` (new
"Results" section) and added to `docs/README.md` index as the "start here" doc.

**Code cleanup (this session) — docstrings + ruff, NO logic changes.** Added top-of-file module docstrings to
32 first-party files (6 `gfs/data/`, 9 `gfs/models/` incl. `antelope*`/`persist`/`transforms`/`get_sampler`,
12 `gfs/trainers/`, `gfs/utils.py`, 4 package `__init__.py`); files that already had docstrings left as-is.
Ran `ruff check --fix` (isort only — `F`/pyflakes NOT in the lint `select`, so NO imports removed) + `ruff
format`, SCOPED to first-party `gfs/` with `--exclude=gfs/models/samplers --exclude=gfs/models/stg` (vendored
third-party left untouched); 25 fixes + 23 files reformatted. **14 `RUF005` "unsafe" fixes intentionally NOT
applied** (`[x]+list`→`[x,*list]`; could change behavior) — `ruff check gfs --exclude=...samplers,...stg`
still reports these 14. Verified: all 47 first-party files compile; key modules import. GOTCHA:
`gfs/models/get_sampler.py` does not import — **pre-existing** missing optional dep `einops` (via vendored
`pps` sampler), confirmed failing identically before the cleanup; unrelated to this work.

## Session log — 2026-09-25

**⚠️ ALL BOOTSTRAPS REDONE (later 2026-09-25) — the seed-level bootstrap p-values in the entries below are
SUPERSEDED.** Re-ran every 4-hop-vs-2-hop comparison with a new **cell-level paired bootstrap** (B=10,000,
`gfs/analysis/paired_cell_bootstrap.py`): one test-cell resample (with replacement) shared across all 5
seeds × both arms, per-seed F1 differenced within seed then averaged; 95% percentile CI + 2-sided p.
Rationale: all cells are from **one biological sample** with a fixed split, so the only resamplable unit is
the cell — but pairing/averaging across the 5 matched seeds is preserved. **Scope: within-sample cell
uncertainty, NOT across-sample generalization; far higher-powered than a seed-level test and can disagree
with it → the paired t-test stays primary.**
- **Consolidated results doc: `docs/20260925_4hop_2hop_cell_bootstrap_summary.md`** (single authoritative
  table for all 11 comparisons × 2 metrics: Δ, boot CI/p, paired-t p, primary call). Observed Δ re-verified
  against every per-doc table (exact match).
- All per-doc bootstrap columns/notes updated to the cell-level values (7 docs:
  `4hop_2hop_comparison`, `20260820_deep_sample_shallow_loss`, `20260919_{abundant_next6,diverse6_balanced,
  diverse6_maxdisp}`, `20260907_subclass_grouping_deep_sample`, `20260914_subclass_grouping_finegrained`).
  `20260811_abundant_subclass` (intact-vs-shuffle, single seed, cell bootstrap B=20k) is a different
  comparison and was left as-is.
- **Read:** the cell-level bootstrap clears α=0.05 on 20/22 rows (only gaba_fine-macro p=0.96 and
  gaba_cge-macro p=0.90 cross zero) — a power/scope artifact. By the **primary paired t-test**, only
  abundant top-6 (both metrics, 5/5), diverse6 **maxdisp-weighted** (t 0.016), and deep_sample
  **Glut-weighted** (t 0.0097) clear α=0.05; the rest are t-ns trends. full22 macro CI is entirely below 0
  (4-hop worse on this sample) yet t-tie — the tie conclusion rests on the t-test.

**diverse6-bal 4-hop vs 2-hop COMPLETE** _(full results: `docs/20260919_diverse6_balanced.md`, Results)_.
Array **`26443774` finished 10/10** (2 arms × 5 seeds, 100 ep + `test_pred.pt`); last run (4-hop seed 4)
done 11:44. This was the only outstanding job — **queue now clear** (user to confirm `squeue`). F1 recomputed
from `test_pred.pt`, paired by seed (n=5, very low power). _(Wilcoxon 2-sided p removed from the docs 2026-09-25
per user request — t-test now primary, bootstrap corroborating.)_
- **weighted** 0.8121 → 0.8322, **Δ+0.0202**, 4/5 seeds; t p=0.121 (primary),
  **bootstrap 2-sided p=0.022** (seed-level paired, B=50k; null-ASL & percentile agree), dz≈0.88.
  → real, consistent-direction gain but **significance is test-dependent → borderline, not a clean win.**
  Bootstrap flags it (leans on the two big +0.04 diffs, seeds 3–4); t-test doesn't; at 5 clusters the
  bootstrap under-covers so its p is optimistic. **t-test primary; bootstrap corroborates direction only.**
- **macro** 0.7524 → 0.7617, +0.0092, 3/5, all tests ns (bootstrap p=0.408, dz≈0.34) — seed 1 −0.028 (053 Sst,
  smallest class). Null on macro.
- Unlike maxdisp (weighted 5/5, t p=0.016), balanced gets **no clean weighted sweep** — seed 0 flips negative;
  the one substitution (030 L6 CT Glut ↔ 333 Endo NN) matters. Panels stable (core Prkcq/Ptprm/Syt6/Zfp536).
- Method note: bootstrap done two ways — seed-level (resample 5 diffs) and cluster (resample seeds + cells);
  identical CI [+0.002,+0.038] weighted, so seed variance dominates (cell resampling negligible at n≈31.6k).
- Also added **bootstrap 2-sided p-values** (seed-level paired, B=50k, null-centered ASL) to the maxdisp & next6
  Results tables for consistency: **maxdisp** weighted <0.001 / macro 0.010 (agrees w/ 5-5 sweep); **next6**
  weighted 0.043 (CI grazes 0, borderline) / macro 0.009. Same caveat everywhere — bootstrap leans on magnitudes,
  under-covers at n=5; t-test stays primary, bootstrap corroborates direction.
- **Bootstrap added to 4 more docs 2026-09-25** (per-seed F1 re-verified against each doc, exact match):
  - `4hop_2hop_comparison.md` (full 22-lbl, n=5): weighted Δ+0.0050 (t 0.45, boot 0.39), macro −0.0067 (t 0.41,
    boot 0.29) — **t & bootstrap fully agree on the null; tie confirmed** (the one arm with no n=5 caveat needed).
  - `20260907_subclass_grouping_deep_sample.md` (Glut/NN/Gaba coarse): Glut wtd boot <0.001 (agrees w/ t=0.0097);
    NN wtd/macro boot 0.036/0.026 & Gaba wtd/macro 0.010/0.002 all dip <0.05 **while primary t is ns** — bootstrap
    over-reads dz≈0.8–1.1 at n=5; t remains the reported ns call.
  - `20260914_subclass_grouping_finegrained.md` (fine/cge/mge): cge-wtd 0.044, mge-wtd 0.037, mge-macro 0.022
    dip <0.05 on bootstrap only; t-test ns everywhere (ceiling). Same under-coverage caveat.
  - `20260811_abundant_subclass.md` (intact vs shuffle, **n=1 seed → cell-level bootstrap**, B=20k): wtd Δ+0.0424
    CI[+0.036,+0.049], macro Δ+0.0511 CI[+0.044,+0.059], both p<0.001, P(Δ>0)=1.0 — intact-graph win solid on
    this run's test set (scope: test-cell sampling for one split, not a seed-level claim).
  - Recurring theme reinforced: **bootstrap is anti-conservative at n=5** (over-flags macro/near-ceiling weighted);
    t-test primary per new convention memory `significance-convention-ttest-primary`.
- No code changed (docs/progress only). Closes the abundant-style diverse-6 family (top6/next6/maxdisp/bal all done).

## Session log — 2026-09-24

**diverse6-maxdisp + abundant-next6 4-hop vs 2-hop COMPLETE** _(full results: `docs/20260919_diverse6_maxdisp.md`,
`docs/20260919_abundant_next6.md`, Results sections)_. Arrays **`26443773` (maxdisp) and `26443761` (next6)
both finished 10/10** (2 arms × 5 seeds, 100 ep + `test_pred.pt`). F1 recomputed from `test_pred.pt` via the
`gaba_grouping_f1.py` helpers (stems `diverse6_maxdisp` / `next6`), paired by seed. n=5 → Wilcoxon floor 0.0625,
so read t-test / dz + sign.
- **maxdisp** — **weighted a real 4-hop win**: 0.8192 → 0.8378, Δ+0.0185, **5/5 seeds, t p=0.016, dz≈1.8**.
  Macro 0.7285 → 0.7510 (+0.0226, 5/5, t p=0.106 ns — gain concentrated in seeds 1 & 3). Strongest abundant-style
  4-hop signal yet; weighted clears significance because this set is below ceiling (~0.82).
- **next6** — positive trend, **not significant**: weighted +0.0082 (4/5, p=0.15), macro +0.0302 (4/5, dz≈0.99,
  p=0.092). seed1 regresses; weighted near ceiling (~0.92). Same direction/magnitude as original top-6 (+0.013/+0.018).
- Panels stable across arms (maxdisp core Grik3/Zfp536/Kcnk9/Slc7a10; next6 core Nxph1/Parm1/Grik3/Sox10).
- **Still running: diverse6-bal array `26443774`** — 2-hop seeds 0–4 done; 4-hop seeds 0–3 mid-train (~ep37–61),
  seed4 (`_9`) was held by the `%4` throttle. **Lifted the throttle 2026-09-24** via
  `scontrol update JobId=26443774 ArrayTaskThrottle=0` (reason flipped `JobArrayTaskLimit`→`None`), so all five
  4-hop seeds now run in parallel — arm ETA ~29 h. Comparison + maxdisp-vs-bal swap pending its 4-hop arm.
  No code changed (docs/progress only).

## Session log — 2026-09-20

**Gaba fine-grained analysis COMPLETE** _(full results: `docs/20260914_subclass_grouping_finegrained.md`, Results section)_.
Array `26380063` finished (all 30/30 runs: fine/cge/mge × 2-hop/4-hop × seeds 0–4, `test_pred.pt` at ep100).
Ran `gfs/analysis/gaba_grouping_f1.py` + per-seed / per-class recall recompute. **Verdict: nothing significant
at n=5.** Weighted at ceiling (0.983–0.993): fine Δ+0.0009, cge +0.0009, mge +0.0005 (all ns). Macro: fine tie,
cge tie (4-hop variance inflated to ±0.055), **mge +0.0541 (dz=0.83, p=0.14 ns)** — driven entirely by rare
laminar MGE types (Sst-Chodl recall +0.19, Pvalb-chandelier +0.11, Lamp5-Lhx6 +0.05). CGE-vs-MGE laminar
contrast points the predicted way (MGE macro +0.054 vs CGE −0.001) but is a trend, not significant. Rare-class
caveat confirmed (macro 0.88 coarse → 0.65–0.81 fine). Panels: fine shows `Nxph1`→`Synpr` swap (as 09-07);
mge both arms converge on identical `{Elfn1,Nxph1,Oprd1,Synpr}`+`Kcnk9`; cge least stable (drops `Igf1` 5→1).
Recorded in doc, this file, and memory `gaba-finegrained-4hop-not-significant`. **This closes doc-2's launched
experiment;** doc-2's deferred groupings (Glut-fine/NN-fine/full-34/type-2) were never built. Doc-1
(`docs/20260907_subclass_grouping_deep_sample.md`) remains fully COMPLETE (runs + analysis).

**Still queued (unchanged, no log dirs yet):** the three abundant-style arrays `26443761` (next-6),
`26443773` (diverse6-maxdisp), `26443774` (diverse6-bal) — see the 2026-09-19 block below; nothing to analyze.
(`squeue` not on assistant PATH — user confirms queue state.) No code changed this session (docs/progress/memory only).

## Current status _(research state as of 2026-09-19)_

**"next-6" abundant-style dataset + experiment scaffolding** _(details: `docs/20260919_abundant_next6.md`)_ —
sibling of the top-6 abundant set, one abundance window down. Established that both the original
`0810_abudant_classes_only_top6` and this new set are **top-N by cell count** (`obs['subclass'].value_counts()`),
NOT from `subclass_edge_pair_counts.csv` (a pairwise spatial-edge table). Built `data/20260919_abundant_next6/`
via `build_subclass_variant.py --keep "327 Oligo NN,005 L5 IT CTX Glut,022 L5 ET CTX Glut,052 Pvalb Gaba,
053 Sst Gaba,032 L5 NP CTX Glut"` (ranks 7–12; 7 labels; from the default 22-label full data, same split as
top-6). First abundant-style set spanning all three families (3 Glut / 2 Gaba / 1 NN). Added configs
`antelope_next6{,_4hop}.yaml` + launcher `scripts/main_next6_all.sh` (array 0-9%4: arm=id/5, seed=id%5) —
same deep-sample protocol (4-hop/tm2 vs 2-hop/tm0, HalfHop-OFF, select-5, 100ep, seeds 0–4). CPU smoke-test
passed (both loaders build; 7 label codes; 4-hop loss_mask=4,627 non-empty). **LAUNCHED — Slurm array
`26443761`** (submitted 2026-09-19; 10 tasks, `%4` throttle). Log dirs `data/logs/*_next6_*_nohh_s*/`; stdout
`outs/out-comms-26443761_<task>.out`. Analysis plan pending runs in the doc.

**Two "diverse-6" arms scaffolded** _(details: `docs/20260919_diverse6_maxdisp.md`, `docs/20260919_diverse6_balanced.md`)_ —
of the top-12 subclasses by abundance, pick 6 by **expression-profile diversity** (centroid dispersion over
500 genes) instead of abundance. Two definitions, one dataset/experiment each:
- **max-dispersion** (`data/20260919_diverse6_maxdisp/`, 1 Glut / 2 Gaba / 3 NN: 007 L2/3 IT, 052 Pvalb,
  053 Sst, 319 Astro, 327 Oligo, 333 Endo) — max total pairwise distance (correlation & Euclidean agree).
- **family-balanced** (`data/20260919_diverse6_balanced/`, 2/2/2: 007 L2/3 IT, 030 L6 CT, 052 Pvalb, 053 Sst,
  319 Astro, 327 Oligo) — max-min dispersion.
Both distinct from top-6 (3/6) and next-6 (3/6); they differ from each other by one subclass (030 L6 CT ↔
333 Endo). Configs `antelope_diverse6_{maxdisp,bal}{,_4hop}.yaml` + launchers `main_diverse6_{maxdisp,bal}_all.sh`
(array 0-9%4). CPU smoke-test passed (both datasets, both loaders; 7 codes; 4-hop loss_mask=4,627). **LAUNCHED
2026-09-19** — max-dispersion = Slurm array `26443773`, family-balanced = `26443774` (10 tasks each, `%4`).
Log dirs `data/logs/*_diverse6_{maxdisp,bal}_*_nohh_s*/`; stdout `outs/out-comms-{26443773,26443774}_<task>.out`.
Analysis pending. (Sibling next-6 arm = array `26443761`.)

**All three new arrays (`26443761` next-6, `26443773` maxdisp, `26443774` bal) are QUEUED, not yet started** —
no log dirs exist for them as of 2026-09-19 (checked); they wait behind the still-running gaba array on the
shared GPU pool and will trickle through (each `%4`). Nothing to analyze for them yet.

**Gaba fine-grained array `26380063` — COMPLETE + ANALYZED (2026-09-20).** All **30/30 runs done**
(fine/cge/mge × 2-hop/4-hop × seeds 0–4, all `test_pred.pt` at ep100; last three mge_4hop seeds 2/3/4 finished
by 2026-09-20 morning). Analysis DONE via `gfs/analysis/gaba_grouping_f1.py` — full write-up in
`docs/20260914_subclass_grouping_finegrained.md` (Results section). **Verdict:** 4-hop is directionally ≥ 2-hop
everywhere (all weighted Δ positive: fine +0.0009, cge +0.0009, mge +0.0005) but **nothing clears α=0.05 at
n=5**; weighted uniformly at ceiling (0.983–0.993, no headroom). Macro splits by set: fine tie (+0.0003), cge
tie (−0.0007, inflated 4-hop variance), **mge +0.0541 (dz=0.83, ns)** — driven entirely by rare laminar types
(Sst-Chodl recall +0.19, chandelier +0.11, Lamp5-Lhx6 +0.05). CGE-vs-MGE laminar contrast points the predicted
way (MGE macro +0.054 vs CGE −0.001) but is a trend, not significant. Rare-class caveat confirmed: macro
dropped 0.88 (09-07 coarse) → 0.65–0.81. Panels: fine shows the `Nxph1`→`Synpr` swap (as 09-07 coarse); mge
both arms converge on identical `{Elfn1,Nxph1,Oprd1,Synpr}`+`Kcnk9`; cge least stable (drops `Igf1` 5→1).

**Next steps (priority order, as of 2026-09-19):**
1. ~~When gaba array `26380063` finishes: run the fine-grained analysis.~~ **DONE 2026-09-20** — see the
   Results section in `docs/20260914_subclass_grouping_finegrained.md` (verdict summarized above). No 4-hop
   result significant at n=5; MGE macro is the one substantive (ns) signal.
2. **Abundant-style arrays: ALL DONE + analyzed.** next-6 (`26443761`) + diverse6-maxdisp (`26443773`) 2026-09-24;
   diverse6-bal (`26443774`) **DONE + analyzed 2026-09-25** (see 09-25 log + `docs/20260919_diverse6_balanced.md`
   Results). maxdisp-vs-bal swap (`030 L6 CT Glut ↔ 333 Endo NN`) done: bal gets no clean weighted sweep (seed 0
   flips) vs maxdisp's 5/5. **Remaining:** the cross-set comparison holding protocol fixed and varying the label
   set — **top-6 / next-6 / diverse-6 max-dispersion / diverse-6 family-balanced**, all on the same split (write-up
   not yet consolidated into one place).
3. All four new arrays share the current-standard deep-sample protocol; only the label set differs (verified via
   config diff — sole non-dataset delta is n_labels, all 7). Metric of record = weighted F1; recompute from
   `test_pred.pt`, never plots. Rare-class caveat: 032 L5 NP / 053 Sst are smallest kept classes → unstable macro.

## Current status _(research state as of 2026-09-14 — Gaba fine-grained run RUNNING)_
**Gaba fine-grained subclass-grouping run (4-hop vs 2-hop on the enriched 34-label set)** — direct follow-up
to the 09-07 experiment, on the new source `data/0914_all_subclasses/` (same cells/graph/genes/split, but 34
labels — fewer subclasses collapsed into `other`; verified clean 50/50 val:test, max |val−test|=1). Motivation:
09-07 coarse Gaba was **ceiling-bound** (only 4 easy canonical classes → 0.984 wtd, no headroom); the enriched
set adds the *hard* rare interneuron subtypes, creating real macro-F1 headroom where a spatial graph should matter.

Three keep-sets built via `gfs/data/build_subclass_variant.py --src 0914_all_subclasses` (new `--src` arg):
- **gaba_fine** (`data/20260914_gaba_fine`, 9 lbl) — all 8 Gaba subclasses + other.
- **gaba_cge** (`data/20260914_gaba_cge`, 4 lbl) — 046 Vip, 047 Sncg, 049 Lamp5 (CGE/superficial lineage).
- **gaba_mge** (`data/20260914_gaba_mge`, 6 lbl) — 050 Lamp5-Lhx6, 051 Pvalb-chandelier, 052 Pvalb, 053 Sst,
  056 Sst-Chodl (MGE/mid-deep lineage). CGE-vs-MGE = a laminar/developmental-origin contrast.
- NB vs 09-07: cge adds Sncg(047); mge adds Lamp5-Lhx6(050), Pvalb-chandelier(051), Sst-Chodl(056) — same cells,
  newly labeled (were folded into `other` at 09-07; 09-07 gaba kept only 046/049/052/053).

Protocol **byte-identical to 09-07** (verified by config diff, only path/n_labels/name differ): 2-hop baseline
`antelope.py` (n_hops=2, tm0) vs 4-hop complete-field `antelope_4hop.py` (n_hops=4, loss_hops=2, tm2),
HalfHop-off, select-5, 100 ep, lr=1e-3 const, best-`val_f1_overall` ckpt. Configs
`antelope_gaba_{fine,cge,mge}[_4hop].yaml`; launcher `scripts/main_gaba_grouping_all.sh` (`--array=0-29%4`,
task→(dataset=id/10 [0=fine,1=cge,2=mge], arm=(id%10)/5, seed=id%5)). CPU smoke-test passed (all 3 load, labels
9/4/6, forward OK, 4-hop loss_mask non-empty). Full detail: `docs/20260914_subclass_grouping_finegrained.md`.

**Running Slurm (2026-09-14 07:36):** array **`26380063`** (`sbatch main_gaba_grouping_all.sh`) — tasks 0–3
(gaba_fine seeds 0–3) training, rest queued behind `%4`. Launched healthy (GPU acquired, 115K-param model,
training underway; `.err` = warnings only). **No `test_pred.pt` yet.** Log dirs
`data/logs/20260914_072*_gaba_fine_*_nohh_s*/`; stdout `outs/out-comms-26380063_<task>.out`.

**Next steps (priority order):**
1. ~~When `26380063` completes: recompute wtd/macro F1 + significance + panels for fine/cge/mge...~~
   **SUPERSEDED — DONE 2026-09-20** (see the 2026-09-20 Session log at the top of this file and the Results
   section in `docs/20260914_subclass_grouping_finegrained.md`).
2. (Deferred) Build Glut-fine (16) / NN-fine (11) / full-34 + type-2 partitions only if the Gaba result warrants.
3. Everything below (the 09-07 experiment) is COMPLETE — see next block.

---

## Current status _(research state as of 2026-09-14)_
**Subclass-grouping deep-sample experiment (4-hop vs 2-hop across Glut / NN / Gaba)** — one throttled array
`25585854` (`main_subclass_grouping_all.sh`, `--array=0-29%4`; task→(dataset,arm,seed)). **COMPLETE — all 30
runs done (3 datasets × 2 arms × 5 seeds).** Gaba 4-hop finished 2026-09-13 23:49 (last task `25585854_29`).
Full design + verified per-seed results + panels + cross-family verdict: `docs/20260907_subclass_grouping_deep_sample.md`.
Families: **Glut** = glutamatergic (excitatory), **NN** = **non-neuronal** (glia + vascular), **Gaba** = GABAergic.

**Verified so far** (recomputed from `test_pred.pt`, n=31,562, population sd):

| dataset | 2-hop wtd | 2-hop macro | 4-hop wtd | 4-hop macro | Δ wtd (sig) |
|---|---|---|---|---|---|
| Glut | 0.8396 ± 0.0037 (n=5) | 0.7369 ± 0.0116 | 0.8494 ± 0.0023 (n=5) | 0.7431 ± 0.0171 | **+0.0098** (5/5, p=0.0097) |
| NN (non-neuronal) | 0.8731 ± 0.0031 (n=5) | 0.5187 ± 0.0384 | 0.8818 ± 0.0096 (n=5) | 0.5544 ± 0.0240 | +0.0087 (**ns**, p=0.15) |
| Gaba | 0.9834 ± 0.0009 (n=5) | 0.8841 ± 0.0080 | 0.9844 ± 0.0003 (n=5) | 0.8931 ± 0.0043 | +0.0010 (**ns**, p=0.088, 4/5); macro +0.0090 (5/5, p=0.066) |

**Findings:** Glut 4-hop beats 2-hop on weighted F1 (+0.0098, all 5/5 seeds, p=0.0097), macro tied. **NN
(non-neuronal) 4-hop trends up (wtd +0.0087 p=0.15, macro +0.0357 p=0.12) but is NOT significant at n=5** —
per-seed diffs flip sign twice; the clean Glut win does not replicate for NN. NN 2-hop has high weighted /
**low macro (0.52)** (rare NN classes missed under collapsed-`other`); 4-hop lifts macro to 0.554 but not
significantly. NN panel backbone = `Sox10`(137)+`Slc7a10`(45); 4-hop scatters the other 3 slots (less stable
than 2-hop). **Gaba 4-hop trends up on both metrics, neither significant at n=5: wtd +0.0010 (4/5, p=0.088,
near ceiling 0.984), macro +0.0090 (5/5 seeds, p=0.066, Wilcoxon at the n=5 floor).** Panel keeps `{Gad2,Elfn1}`
backbone and swaps `Nxph1`→`Synpr`.
**Cross-family verdict (all 6 dataset×metric Δ positive — 4-hop never loses on the mean, but only Glut-weighted
clears α=0.05; NN never does, Gaba's macro sits at the n=5 Wilcoxon floor).** Consistent with
`4hop-does-not-beat-2hop-full-data`: the deep-sample win is real in direction but small and rarely significant
across label sets at n=5. See the doc's Cross-family verdict block.

**Running Slurm (2026-09-14):** none for this experiment — array `25585854` fully complete (last task finished
2026-09-13 23:49; no fresh event files since).

**Next steps (priority order):**
1. Experiment complete. If a stronger significance claim is wanted for NN/Gaba, rerun with more seeds
   (n=5 leaves Gaba-macro & NN at the Wilcoxon floor); Glut-weighted (p=0.0097) already stands.
2. NN 4-hop seed0 = restart dir `20260910_222140` (stale `20260909_173231` @ep50 ignored); seed4 = `20260910_193926`.

**Done this session (2026-09-13):** recomputed NN (non-neuronal) 4-hop vs 2-hop F1 (n=5), paired
significance tests (weighted t=1.78 p=0.15; macro t=1.98 p=0.12; both ns), and panel comparison; recorded in
`docs/20260907_subclass_grouping_deep_sample.md`.

Prior status below.

## Current status _(research state as of 2026-09-05)_
**GR2ST attention-entropy λ sweep COMPLETE — all 4 arms done (seed 0, 4-hop/100ep/select-5/HalfHop-OFF, identical splits).**
Test F1 recomputed from `test_pred.pt` (n=31,562):

| λ | wtd F1 | macro F1 | panel (gene idx) | val eff-neigh | entropy | top1 |
|---|---|---|---|---|---|---|
| 0.0 | 0.6849 | 0.5056 | {2,83,225,250,480} | 16.9 | 0.400 | 0.122 |
| 0.001 | **0.7047** | **0.5378** | {2,12,83,225,480} | 16.7 | 0.398 | 0.126 |
| 0.01 | **0.7052** | 0.5309 | {2,12,83,225,480} | 15.5 | 0.387 | 0.140 |
| 0.05 | 0.6375 | 0.4624 | {2,83,435,480} | 5.5 | 0.147 | 0.669 |

**Verdict:** gentle entropy pressure (λ=0.001–0.01) improves F1 (+0.020 w) and recovers the stable panel
{2,12,83,225,480} (gene 12 back in); over-sharpening (λ=0.05) collapses attention (eff-neigh→5.5, top1→0.67)
and *drops F1 below baseline* (0.6375 w) with the panel drifting off-backbone. Useful window ≈ 0.001–0.01;
**no need to extend to 0.1** (0.05 already fails the "sharpen WHILE F1 holds" criterion). Recommend λ=0.01.
λ=0.0 reproduces the 4-hop seed-0 run exactly (sanity check). See `docs/20260902_gr2st_attention_entropy.md`.
Prior status below.

## Current status _(research state as of 2026-08-25)_
HalfHop 2x2 ablation complete for seed 0 (`docs/20260820_deep_sample_shallow_loss.md`); **HalfHop-OFF
comparison now has 5-seed error bars (seeds 0–4)**. Grid (seed-0, weighted / macro F1, recomputed from
`test_pred.pt`, n=31,563):

| | HalfHop ON | HalfHop OFF |
|---|---|---|
| 2-hop / trainmode0 (baseline) | **0.7789** / 0.7618 | 0.7750 / 0.7603 |
| 4-hop / trainmode2 (complete-field loss) | 0.7671 / 0.7563 | **0.7933 / 0.7824** |

**HalfHop-OFF multi-seed (0–4), added 2026-08-25** (arrays 2hop=`25336508`, 4hop=`25336509`; recomputed from
`test_pred.pt`, configs verified from `hparams.yaml`):

| Arm (HalfHop OFF) | weighted (mean ± sd) | macro (mean ± sd) |
|---|---|---|
| 2-hop / tm0 | 0.7719 ± 0.0043 | 0.7583 ± 0.0054 |
| 4-hop / tm2 | **0.7851 ± 0.0073** | **0.7764 ± 0.0067** |

**Findings:** (1) The 4-hop complete-receptive-field gain **replicates on all 5 seeds** (4hop > 2hop every
seed; +0.0132 w / +0.0181 m, gap > combined spread). (2) [seed-0 only] HalfHop's sign flips with sampling
depth: barely helps at 2-hop (+0.004), *hurts* at 4-hop (−0.026). **Panels (multi-seed, checkpoint epoch):**
backbone `A830036E02Rik`(83, 10/10), `Hmcn1`(325, 9/10), `Dscaml1`(480, 8/10); depth signature = `Syt6`(5)
in all 5 4-hop seeds vs `Grik3`(2) dominant at 2-hop (see doc panel tables). HalfHop-ON seeds 1–4 still not run.

Prior context: abundant-subclass seed-0 experiment (`docs/20260811_abundant_subclass.md`) — selection is
graph-invariant (both arms picked `{2,12,38,83,480}`); intact graph helps classification +0.042 w / +0.051 m.

### 2026-08-30 — Full-data 4-hop vs 2-hop comparison scaffolded _(details: `docs/4hop_2hop_comparison.md`)_
- New experiment: replicate the abundant 4-hop > 2-hop win on the **full 22-label** data (GFS select-5,
  HalfHop off, seeds 0–4, 10 runs). Same diagonal as the abundant 2×2: 2-hop `trainmode=0` (loss on all
  patch nodes) vs 4-hop `n_hops=4`/`trainmode=2` (loss on complete-field nodes only); message passing 2 hops
  and root-only eval in both.
- Added `gfs/configs/antelope_4hop.yaml` (full-data 4-hop, `n_labels=22`), `scripts/main_fulldata_2hop_seeds.sh`
  and `scripts/main_fulldata_4hop_seeds.sh` (both Slurm array 0–4). Configs parse-checked; not yet launched.

### 2026-08-31 — PERSIST-panel no-graph baseline scaffolded _(details: `docs/baseline_comparisons.md`)_
- New experiment to give the regenerated PERSIST select-5 panels **test metrics** (PERSIST computes none)
  and isolate the **graph's contribution**: score each PERSIST panel with the **self-edges (no-graph)**
  `LitGnnFs` and compare to the GFS **full-graph 4-hop** model (job 25507593; reference pending — arm still
  diverging). _[reference arm switched 2-hop→4-hop 2026-08-31, see later entry.]_
- Added `--no-upsample` to `gfs/data/node_exposure.py` (plain 1× tabular train, skips exposure counting;
  still writes `upsampled_train.h5ad`/`tabular_{val,test}.h5ad`/`classes.json` so `antelope_panel.py` +
  `PanelTabularDataModule` consume it unchanged). Built `exposure/persist_1x_seed{0-4}` (74,633 train /
  31,562 val / 31,562 test, 5 genes, 22 classes each) — one per PERSIST panel.
- Added `scripts/main_panel_persist_1x.sh` (array 0–4, task id → panel index, `rand_seed=0` all runs,
  `batch_size=512` → ~146 steps/epoch on the 1× train). Smoke-tested: panels verified against source
  var_names, datamodule builds, batch x(512,8) self-loop edges, labels 0–21. **Not yet launched.**
- Added a **shuffled-graph arm** (per user request): PERSIST panel + GFS model over a shuffled graph
  (`antelope_panel_graph.py`, `shuffle_cells=true`) — no real neighbour signal, but the 2-hop
  NeighborLoader gives ~65×/cell exposure by construction (the "essentially upsampling + noisy conv" arm,
  between the self-edges floor and the intact graph). `scripts/main_panel_graph_persist_shuffle.sh` (array
  0–4, `rand_seed=0`, **`halfhop=false`** to match the GFS reference / avoid eval-HalfHop). Smoke-tested
  (datamodule builds, shuffles train+valtest, patch x(3997,8) from 64 seeds). **Not yet launched.**

### 2026-08-31 (later) — baseline_comparisons GFS reference switched 2-hop → 4-hop _(doc only)_
- Per user decision, the **GFS + full graph** reference arm in `docs/4hop_2hop_comparison.md`'s companion
  `docs/baseline_comparisons.md` is now the full-data **4-hop** arm (`antelope_4hop.py`/`antelope_4hop.yaml`,
  job **25507593**), replacing the 2-hop arm (25507587) throughout that doc.
- Rationale (user): the 4-hop arm is directly comparable to the 2-hop frozen-panel arms because
  `trainmode=2` takes loss on the **same within-2-hop node set** (2-layer message passing in both); the
  extra hops 3–4 only complete receptive fields, so per-cell loss exposure is unchanged. Removed the earlier
  "sampling-depth mismatch" caveat as overstated.
- **Consequence:** the reference numbers were pending while the 4-hop arm ran. **Resolved 2026-09-01** —
  the 4-hop arm (25507593) completed; reference = 0.7099 ± 0.0142 wtd / 0.5272 ± 0.0141 macro, now filled
  into `docs/baseline_comparisons.md`. The 2-hop arm remains the baseline in `docs/4hop_2hop_comparison.md`.

## In progress / running jobs
_As of 2026-11-02 the Slurm queue is CLEAR — no jobs newer than `27095776` (finished 2026-09-28); `outs/`
confirms. User to verify with `squeue -u donghyun.kim`. Entries below are historical status._
- **DONE 2026-09-20, job `26380063`** — **Gaba fine-grained grouping** (`docs/20260914_subclass_grouping_finegrained.md`;
  all 30 runs complete, analyzed — verdict: nothing significant at n=5, MGE macro +0.054 ns the one signal;
  memory `gaba-finegrained-4hop-not-significant`). Fuller Glut-16 / NN-11 / full-34 groupings still PLANNING (not built):
  built off the user's new enriched label set `data/0914_all_subclasses/` (34 labels; less collapsed into
  `other`; split verified clean 50/50). Three Gaba keep-sets (fine subclasses + `other`), 2-hop vs 4-hop,
  seeds 0–4: **`20260914_gaba_fine`** (all 8 Gaba, 9 lbl), **`20260914_gaba_cge`** (Vip/Sncg/Lamp5, 4 lbl),
  **`20260914_gaba_mge`** (Pvalb/Pvalb-chandelier/Sst/Sst-Chodl/Lamp5-Lhx6, 6 lbl). Motivation: 09-07 Gaba was
  ceiling-bound (0.984 wtd, 4-hop not sig); enriched subtypes add real macro headroom + CGE-vs-MGE laminar
  contrast. Datasets built (builder gained `--src`), 6 configs + one throttled array
  `scripts/main_gaba_grouping_all.sh` (`--array=0-29%4`, task id → dataset/arm/seed: ds=id/10 [0=fine,1=cge,
  2=mge], arm=(id%10)/5, seed=id%5). CPU smoke-tested (loaders/labels/forward/loss_mask all pass), ruff clean.
  Launched via `main_gaba_grouping_all.sh`, finished + analyzed 2026-09-20 (F1 + paired significance + panels
  written to the finegrained doc; compared vs 09-07 coarse Gaba there).
- **DONE 2026-09-14** — **subclass-grouping deep_sample variants** (`docs/20260907_subclass_grouping_deep_sample.md`):
  ran as one throttled array **`25585854`** (`main_subclass_grouping_all.sh`, `--array=0-29%4`, ≤4
  concurrent; task id → dataset/arm/seed). **All 30 runs complete** (last task finished 2026-09-13 23:49);
  F1 + paired significance + panels recomputed and written to the doc (see Cross-family verdict).
  Three new datasets built from the fine-stratified full
  data via `gfs/data/build_subclass_variant.py` — `20260907_glut_classes` (10 lbl), `20260907_nn_classes`
  (8 lbl), `20260907_gaba_classes` (5 lbl); each keeps a subclass family + `other`, same cells/graph/masks as
  full data (no leakage, 50/50 stratified val:test verified, all within 1 cell). 6 dedicated configs
  (`antelope_{glut,nn,gaba}[_4hop].yaml`) + the combined array launcher `main_subclass_grouping_all.sh`
  (also 6 per-arm scripts `main_{glut,nn,gaba}_{2hop,4hop}_seeds.sh` as a fallback). All 30 runs launched via
  the combined array above. **Time limit 60 h/task is sufficient** — the 4-hop arm is the binding constraint
  at ~29 h (measured from the completed abundant 4-hop runs; identical graph/batch/epochs), 2-hop ~6 h. On
  completion: recompute F1 from `test_pred.pt`, fill the doc's per-dataset table. (Glut drops subclass 020 —
  0 valtest cells.)
- **DONE 2026-09-28 (analyzed 2026-11-02), job `27095776`** — **feature-ablation ABUNDANT frozen-panel arm**
  (`docs/feature_ablation_abundant.md`): array tasks 0–4, task id → PERSIST panel index, `rand_seed=0` all
  runs, HalfHop off, `fs_method=none` (frozen 5-gene panel, no selection). Launcher
  `scripts/main_abundant_4hop_panel_seeds.sh`. Log dirs
  `data/logs/20260927_11145{2,4}_abundant_4hop_panel_panel{0-4}_nohh_s0/` (panel1 at `_111454`, rest `_111452`).
  F1 recomputed from each `test_pred.pt` (n=31,563, 7 classes; weighted == logged `test_f1_overall`): frozen
  **0.7791±0.0112 w / 0.7710±0.0165 m** vs selection **0.7851±0.0073 / 0.7764±0.0067** → +0.0060 w / +0.0054 m.
  Cell-level bootstrap sig (tiny), seed-level ns. Doc Results/Significance/Findings filled, Runs table marked
  done. With-selection reference reused (`data/logs/20260820_182036_abundant_4hop_seed0_nohh_s0` + seeds 1–4).
  Prereq abundant PERSIST panels **DONE 2026-09-07** (jobs 25585849–853 → `~/persist/results/persist_panel_select5_seed{0-4}_abundant.csv`,
  `exposure=none`, non-upsampled): s0 [2,167,185,205,480], s1 [1,2,83,416,480], s2 [2,5,167,205,480],
  s3 [2,167,185,435,480], s4 [2,38,167,435,480].
- **DONE 2026-09-04** — **GR2ST attention-entropy on/off comparison** (`docs/20260902_gr2st_attention_entropy.md`):
  Slurm array **25553166** (`--array=0-1`, both `rand_seed=0`). Both 4-hop / 100ep / select-5 / HalfHop-OFF
  arms finished. **task 0** `25553166_0` (λ=0.0, `log_attention_stats=true`): **0.6849 w / 0.5056 m**, panel
  {2,83,225,250,480} — reproduces the 4-hop seed-0 run exactly (sanity check). **task 1** `25553166_1` (λ=0.01,
  GR2ST default): **0.7052 w / 0.5309 m**, panel {2,12,83,225,480}. **Success criterion met + F1 improved**
  (+0.020 w / +0.025 m) while sharpening attention (ep99 val eff-neighbors 16.9→15.5, entropy 0.400→0.387,
  top1 0.122→0.140) — but the ep99 sharpening is far milder than the ep30 mid-anneal peek (tau re-diffuses
  attention as selection commits, so the ep30 6.9 eff-neighbors was NOT the final state). λ=0.01's panel is
  identical to the 2-hop seed-0 panel — it recovers gene 12 that plain 4-hop seed-0 dropped (cross-ref filled
  into `docs/4hop_2hop_comparison.md`). Docs updated.
- **DONE 2026-09-05** — **GR2ST attention-entropy λ=0.001 arm** (`scripts/main_4hop_gr2st_ae0p001.sh`, job
  **25555463**, `full_4hop100_ae0p001_nohh_s0`): **0.7047 w / 0.5378 m**, panel {2,12,83,225,480}. Ties λ=0.01
  on weighted, best macro of the sweep; attention barely sharpens at this weight (ep99 eff-neigh 16.7).
- **DONE 2026-09-05** — **GR2ST attention-entropy λ=0.05 arm** (`scripts/main_4hop_gr2st_ae0p05.sh`, job
  **25555510**, `full_4hop100_ae0p05_nohh_s0`): **0.6375 w / 0.4624 m**, panel {2,83,435,480}. Over-sharpens
  (ep99 eff-neigh 5.5, top1 0.669) and F1 drops below the λ=0 baseline → brackets the top of the useful window.
  Sweep complete; verdict + full table in the Current status block above and `docs/20260902_gr2st_attention_entropy.md`.
- **CANCELLED 2026-09-02** — **300-epoch probe** of the 4hop-vs-2hop full-data comparison
  (`docs/4hop_2hop_comparison.md`): 2-hop = job **25522891** (both seeds ✅ done — seed 0 **0.7006 wtd /
  0.5054 macro**, seed 1 **0.7114 / 0.5249**, panel {2,12,83,435,480}; longer tau exploration did NOT help,
  matches 100ep 2-hop). **4-hop = job 25522893 CANCELLED** — measured ~17 min/epoch → ~85h for 300ep, over the
  60h `--time` cap; would have been killed at ~ep211/215, **mid-tau-anneal** (tau≈0.078, selection uncommitted)
  → partial checkpoints unusable. No 4-hop 300ep results. **TO RESUME LATER via checkpoint-resume** (requeue
  across 60h blocks; needs `trainer.fit(ckpt_path=...)` + Slurm `--requeue` added to `antelope_4hop.py`).
  Low priority — the 100ep result already answers 4hop-vs-2hop; this only tested longer exploration.
- **DONE 2026-09-01** — full-data **4-hop** arm `25507593` (seeds 0–4, `main_fulldata_4hop_seeds.sh`,
  22-label, GFS select-5, HalfHop off) finished all 5 seeds (100 ep). Recomputed F1 = logged
  `test_f1_overall`: **0.7099 ± 0.0142 wtd / 0.5272 ± 0.0141 macro** — statistically **tied** with the
  2-hop arm `25507587` (0.7049 ± 0.0070 / 0.5339 ± 0.0061). **The abundant-subset 4hop>2hop win does NOT
  replicate on full 22-label data.** The 2026-08-31 "divergence" was a false alarm (transient mid-tau-anneal
  snapshot at ep~20; by ep99 tau annealed, selection committed to probs≈0.999, test_loss_ce 0.85–0.94). This
  arm is now the **completed GFS full-graph reference** for `docs/baseline_comparisons.md`. Docs updated
  (`4hop_2hop_comparison.md`, `baseline_comparisons.md`). Panels: core {2,83,480} all 5 seeds + 12 in 4/5.
- **DONE 2026-09-02** — feature-ablation frozen-panel arm `25511099` (`docs/feature_ablation.md`;
  `--array=0-4`, task id → PERSIST panel index, all `rand_seed=0`): 4-hop GFS `fs_method=none` frozen to the
  regenerated PERSIST select-5 panels (`scripts/main_fulldata_4hop_panel_seeds.sh`). **All 5 panels finished**
  (per-panel wtd/macro: 0→0.7111/0.5302, 1→0.6430/0.5184, 2→0.6907/0.5095, 3→0.6835/0.5262, 4→0.6950/0.5146;
  recomputed from `test_pred.pt`, matches logged `test_f1_overall`). **Frozen arm = 0.6847 ± 0.0254 wtd /
  0.5198 ± 0.0084 macro** vs with-selection **0.7099 ± 0.0142 / 0.5272 ± 0.0141** (job 25507593): selector
  wins **+0.025 wtd / +0.007 macro**, within ~1 combined sd. The frozen arm's deficit is variance not ceiling
  — 4/5 panels land 0.68–0.71 (panel 0 = 0.7111 *beats* selector seed-0), only panel 1 ({2,25,83,207,225},
  missing the {12,480} backbone) drops to 0.6430. Read: most of the 4-hop F1 is "a good panel + graph
  backbone," not end-to-end selection. Docs (`feature_ablation.md`) filled: results table, summary, findings.
- **DONE 2026-08-31** — regenerated the feature-ablation frozen panels with the **pristine upstream PERSIST**
  (jobs `25507771`–`25507775`, non-upsampled, post label-fix) → `persist/results/persist_panel_select5_seed{0-4}.csv`.
  Panels swapped into the `case` blocks and filled into `docs/feature_ablation.md` + `docs/baseline_comparisons.md`.
  Supersedes the compromised (deleted) `persist_genes_selected_genes.csv` (memory `persist-panels-regenerated-not-stale-csv`).
- **READY, not launched 2026-08-31** — PERSIST-panel no-graph baseline (`docs/baseline_comparisons.md`):
  self-edges `LitGnnFs` on the 5 regenerated PERSIST panels, `rand_seed=0` all runs, non-upsampled 1×
  (`scripts/main_panel_persist_1x.sh`, array 0–4). Datasets built (`exposure/persist_1x_seed{0-4}`),
  smoke-tested. On completion: recompute weighted+macro F1 from `test_pred.pt`, fill the results table vs
  the GFS full-graph **4-hop** arm (25507593, reference pending). **Second arm** (shuffled graph): `main_panel_graph_persist_shuffle.sh`
  (array 0–4, `halfhop=false`) — same doc, fills the shuf-graph columns.
- **DONE 2026-08-25** — `25336508` (2-hop nohh seeds 1-4) and `25336509` (4-hop nohh seeds 1-4) both
  finished; F1 recomputed per seed from `test_pred.pt` and folded into the multi-seed table above.

**Running as of 2026-09-05** (recheck via `selections.csv` row count / tfevents `epoch` scalar — the
`.out`/`.err` files only show the model summary; Lightning's progress bar isn't flushed to file. Logs are
TensorBoard event files only, no `metrics.csv` — read scalars with `EventAccumulator`):
- **GR2ST attention-entropy λ sweep — ✅ ALL 4 ARMS DONE** (λ=0.0 `25553166_0`, 0.001 `25555463`, 0.01
  `25553166_1`, 0.05 `25555510`); results table + verdict in Current status. No GR2ST jobs still running.
- 300ep probe `25522893` (4-hop) **CANCELLED** (wall-clock); 2-hop `25522891` both seeds done.
- feature-ablation `25511099`: **all 5 panels ✅ done**.
- **Not launched** (user must `sbatch`): baseline_comparisons arms `main_panel_persist_1x.sh` +
  `main_panel_graph_persist_shuffle.sh`; HalfHop **ON** abundant seeds 1–4.

## Next steps (priority order)
0-new (2026-11-02, handoff). **Queue is clear — no experiment is running or pending launch.** Open items:
   (a) **Uncommitted repo state is large and intentional — do NOT bulk-commit.** The entire `docs/` tree and
   `CLAUDE.md` are untracked; nearly all of `gfs/` is modified (this session's docstrings + ruff format, ON TOP
   OF pre-existing uncommitted edits to `antelope.py`, `hemisphere_new.py`, `trainers/antelope.py`,
   `configs/antelope.yaml`, and the `scripts/`/`scripts_0505/` deletions — none of those pre-existing changes
   are from this session). Commit deliberately in logical groups only when the user asks.
   (b) Optional: apply the 14 `RUF005` unsafe ruff fixes (`ruff check gfs --exclude=gfs/models/samplers
   --exclude=gfs/models/stg --fix --unsafe-fixes`) — left undone to avoid any behavior change.
   (c) `gfs/models/get_sampler.py` + `antelope_topk*` need `einops` (not installed); `pip install einops` if
   the top-k samplers are ever used.
   (d) Verified-feature-ablation prediction (from `docs/conclusion.md`): redo claims 1.1/1.2 with the 4-hop
   selector panel (currently use the 2-hop panel `[2,12,83,435,480]`). Low priority; expected to hold.
0-done (2026-09-27→11-02). **Feature-ablation ABUNDANT frozen-panel arm — DONE** (job `27095776`, analyzed
   2026-11-02; `docs/feature_ablation_abundant.md` complete). Note for reference: `docs/feature_ablation.md`
   full-data significance was computed ad-hoc in scipy from the doc's per-run table (cell-level bootstrap via
   the committed `gfs/analysis/feature_ablation_cell_bootstrap.py`; the seed-level paired t has no committed
   script).
0-new (2026-09-25). **Bootstrap redo — DONE this session; no compute pending.** Only follow-ups:
   (a) the new analysis script `gfs/analysis/paired_cell_bootstrap.py` and all `docs/*.md` are **untracked/
   uncommitted** (the whole repo has a large pre-existing uncommitted state — do NOT bulk-commit; commit the
   bootstrap script + updated docs deliberately if/when the user asks).
   (b) Optional: fold Cohen dz + per-seed sign columns into the consolidated table
   `docs/20260925_4hop_2hop_cell_bootstrap_summary.md` (values already in each per-doc table; I left it lean).
   (c) The consolidated bootstrap run output is saved at `outs/paired_cell_bootstrap.log`; rerun with
   `python -m gfs.analysis.paired_cell_bootstrap` (gfs env, ~15-20 min, CPU-only, no Slurm).
0a. **Subclass-grouping array `25585854` — DONE 2026-09-14** (`docs/20260907_subclass_grouping_deep_sample.md`):
   weighted + macro F1 recomputed from all 30 `test_pred.pt`, paired t + Wilcoxon + Cohen dz per dataset,
   panels + cross-family verdict written to the doc. Verdict: all 6 dataset×metric Δ positive, but only
   Glut-weighted significant (p=0.0097); NN ns (p=0.15/0.12), Gaba trends up (macro 5/5, p=0.066; wtd near
   ceiling). Only remaining option if a firmer NN/Gaba claim is wanted: more seeds (n=5 hits the Wilcoxon floor).
0a-follow-up. **New subclass groupings to extend the deep-sample experiment** (discussed 2026-09-14, nothing
   built/launched yet). The suffix-family axis is EXHAUSTED — Glut/NN/Gaba are the only 3 families and all are
   done. Full data = 22 subclasses: Glut ×10 (~44k), NN ×7 (~22k), Gaba ×4 (~5.8k), + other. New groupings
   need explicit `--keep` in `gfs/data/build_subclass_variant.py`. Options, best-first:
   1. **(recommended) Within-Glut projection classes** — IT (004/005/006/007 = L6/L5/L4-5/L2-3 IT) vs
      deep-projection (022 L5 ET, 030 L6 CT, 029 L6b, 032 L5 NP) vs 001 Car3. Most spatially-laminar axis →
      strongest test of the deeper receptive field; builds on Glut being the one family where 4-hop won (wtd
      p=0.0097). **Always drop 020 (0 valtest cells).**
   2. Cross-family coarse cuts (neurons=Glut+Gaba vs NN vs other; or excitatory-vs-inhibitory) — but these are
      near-ceiling like Gaba (0.984), likely just re-confirm "no headroom"; lower value.
   3. Rare-class stress panel (sub-1k subclasses: 046 Vip, 330 VLMC, 326 OPC, 029 L6b) vs the historical
      abundant top-6 reference.
   To pursue #1: `python -m gfs.data.build_subclass_variant --keep "<list>" --out 20260914_glut_proj`, add a
   config + reuse `main_subclass_grouping_all.sh`-style array (2-hop/4-hop × 5 seeds), same nohh protocol.
0b. **Abundant PERSIST panels `25585849–53` — ON COMPLETION** (`docs/feature_ablation_abundant.md`): paste the
   5 panels from `~/persist/results/persist_panel_select5_seed{0-4}_abundant.csv` into the TODO placeholders in
   `scripts/main_abundant_4hop_panel_seeds.sh` + the doc panel table, then user `sbatch` the frozen-panel array;
   compare to the reused with-selection reference (0.7851 w / 0.7764 m).
1. **GR2ST λ sweep — DONE** (`docs/20260902_gr2st_attention_entropy.md`). Verdict: useful window λ≈0.001–0.01
   (+0.020 w, recovers panel {2,12,83,225,480}); λ=0.05 over-sharpens and drops F1 below baseline, so 0.1 is
   NOT worth running. Recommend λ=0.01. Remaining follow-ups if pursued: (a) multi-seed error bars on λ=0.01 vs
   0.0 (sweep is seed-0 only, so the +0.020 gain lacks error bars); (b) a λ=0.03 point to refine the 0.01→0.05 knee.
2. **Launch the baseline_comparisons arms** (`docs/baseline_comparisons.md`) — user must `sbatch`:
   `main_panel_persist_1x.sh` (self-edges, no graph) and `main_panel_graph_persist_shuffle.sh` (shuffled graph).
   Both array 0–4, `rand_seed=0`, datasets built (`exposure/persist_1x_seed{0-4}`) + smoke-tested. On completion:
   recompute F1, fill the tables, compare to the GFS full-graph 4-hop reference (25507593, 0.7099 ± 0.0142 wtd).
3. **HalfHop ON abundant seeds 1–4** (2-hop `main_abundant.sh`, 4-hop `main_abundant_4hop.sh`) to complete the
   2×2 error bars on the HalfHop sign-flip; check the backbone `{83,325,480}` (+Syt6/Grik3 depth signature) holds.
4. Gate HalfHop to train-only (currently active at eval too) to separate train- vs eval-time contribution —
   ON arms carry perturbed-graph eval, OFF arms raw-graph eval, so the current flip conflates the two.
5. Add the `xyz_status=false` control — some of the intact gain may be "xyz became usable" not message passing.
6. **(Deferred, low priority) Resume the 4-hop 300ep probe** (`docs/4hop_2hop_comparison.md`): needs
   checkpoint-resume support in `antelope_4hop.py` (`trainer.fit(ckpt_path=...)` + Slurm `--requeue`) so 300
   epochs (~85h) complete across multiple 60h blocks. Only tests the longer-tau-exploration hypothesis; the
   100ep result already answers 4hop-vs-2hop, so this is not on the critical path.

## Known blockers / issues
- Slurm CLI unavailable from the assistant's non-interactive shell; user must run `sbatch`/`squeue`/check outputs.
- `self_loops_only=True` is broken in this repo's loader.

## Framing & key decisions
**Framing rule (load-bearing):** never say "the graph is irrelevant" unqualified — true for *selection*,
false for the *classifier*. The graph question splits into two contributions with opposite answers
(selection is graph-invariant; classification gains +0.082 weighted F1 at a matched panel). Full reasoning
trail in `docs/does-the-graph-help.md`.

Decisions for the current experiment:
- Restrict to top-6 abundant subclasses + `other` (7 labels) to test whether the graph shifts *selection*.
- Reuse existing trainers (`antelope.py` / `antelope_shuffle.py`) via Hydra `--config-name`; only the config + sbatch scripts are new.
- Matched pair at `n_select=5, seed=0` so default-vs-shuffle is directly comparable.

Open caveats (from `does-the-graph-help.md`): the +0.082 intact arm still runs `xyz_status=true` (needs an
`xyz_status=false` control); one PERSIST label-fix control outstanding; error-bar seeds 1–4 not yet run.

## Development log

### 2026-09-07 (later) — Per-class accuracy analysis + reusable script _(details: `docs/20260820_deep_sample_shallow_loss.md`)_
- **Per-class accuracy (recall) table added** to the deep_sample doc, comparing 2-hop vs 4-hop (HalfHop-OFF,
  seeds 0–4, mean ± sd, recomputed from `test_pred.pt`, n=31,563). **Finding: the 4-hop win is concentrated in
  the deep-layer L6 Glut classes** — `004 L6 IT` +0.101 (weakest class), `030 L6 CT` +0.057; the other 4
  classes drift −0.004…−0.024 (all within ~1 sd). Balanced acc +0.017 (0.754→0.771). Dovetails with the panel
  result (4-hop swaps in `Syt6`, an L6 marker). Total support = 31,563 ✓.
- **New reusable script `gfs/analysis/per_class_accuracy.py`** (+ `gfs/analysis/__init__.py`, new package dir):
  recomputes per-class recall from `test_pred.pt` across seeds, mean ± sd, prints the markdown table. Class
  order from the dataset's train h5ad `subclass` categories. `ARMS`/`DATA_SUBDIR` hard-coded to the abundant
  deep_sample runs (reproduces the doc table exactly — verified); edit both for other experiments. ruff clean.
- **Time-limit check** on the running subclass-grouping array `25585854`: 60 h/task is sufficient — 4-hop arm
  ~29 h (measured from completed abundant 4-hop runs, identical graph/batch/100-ep), 2-hop ~6 h.
- **Statistical-significance test (DONE 2026-09-14):** seeds matched across 2-hop/4-hop within a dataset (same
  `rand_seed`, identical test cells) → **paired test per dataset**: paired t + Wilcoxon signed-rank + Cohen dz
  on the 5 paired Δ, per metric. Results in the doc's per-dataset significance blocks + Cross-family verdict.
  Note n=5 pins Wilcoxon's smallest attainable p at 0.0625 (all-same-sign), so an all-5/5 win (Glut-wtd,
  Gaba-macro) can only reach significance via the paired t; Glut-wtd does (p=0.0097), Gaba-macro is borderline
  (p=0.066). `scipy` in the `gfs` env.
- All session artifacts are **on disk but uncommitted** (whole repo is uncommitted — see git status).

### 2026-09-04 — GR2ST λ=0.0 & λ=0.01 arms complete _(details: `docs/20260902_gr2st_attention_entropy.md`, `docs/4hop_2hop_comparison.md`)_
- Job progress check: array `25553166` both tasks finished (100 ep, tested from best-val ckpt). Recomputed
  test F1 from `test_pred.pt` (n=31,562): **λ=0.0 = 0.6849 w / 0.5056 m** (panel {2,83,225,250,480});
  **λ=0.01 = 0.7052 w / 0.5309 m** (panel {2,12,83,225,480}). λ=0.001 (`25555463`, ~ep82) and λ=0.05
  (`25555510`, ~ep80) still running, both advancing normally (~5–6 h to finish).
- **Findings:** (1) λ=0.0 reproduces the main-comparison 4-hop seed-0 run *exactly* (F1 + panel) → entropy
  machinery inert at λ=0, sanity check passes. (2) **λ=0.01 met the success criterion AND improved F1**
  (+0.020 w / +0.025 m) while sharpening attention — but only *modestly* at ep99 (val eff-neighbors
  16.9→15.5, entropy 0.400→0.387); the strong ep30 sharpening (eff-neighbors 6.9) **re-diffuses** as tau
  anneals and selection commits, so the mid-anneal peek was NOT the final state. (3) λ=0.01's panel is
  *identical* to the 2-hop seed-0 panel {2,12,83,225,480} — it recovers gene 12 that plain 4-hop seed-0
  uniquely dropped, i.e. attention-sharpening nudged the noisy 4-hop seed 0 back onto the stable panel.
- Docs: filled Results section in `20260902_gr2st_attention_entropy.md` (status table → 2 done / 2 running);
  added a cross-reference table to `4hop_2hop_comparison.md` (seed-0 comparison vs 2-hop/4-hop). No code changed.

### 2026-09-03 — GR2ST attention-entropy λ sweep extended to {0.0, 0.001, 0.01, 0.05} _(details: `docs/20260902_gr2st_attention_entropy.md`)_
- Checked the running on/off comparison `25553166` (@ep30/100, mid-anneal). **Mechanism confirmed:** λ=0.01
  sharpens attention strongly — val effective-neighbors 14.2→6.9, entropy 0.359→0.246, top1-mass 0.134→0.334,
  normalized-entropy 0.820→0.524 — while the weighted loss term stays tiny (~0.002). `val_f1` still ~0.08–0.12
  on all arms: expected mid-anneal undertraining (tau≈1.45), NOT a result. Panels @ep25–29 are **identical**
  across the 0.0 and 0.01 arms: `{324,105,472,280,228}` (gene indices) — entropy loss hasn't changed *which*
  genes are picked so far (mid-anneal, can still shift). Verified timing: ~17.3 min/ep → ~28.5 h to ep100, well
  under the 60 h cap.
- Added two weight points to complete the sweep (single jobs, seed 0, everything else identical to the on/off arms):
  - `scripts/main_4hop_gr2st_ae0p001.sh` → job **25555463**, λ=0.001, prefix `full_4hop100_ae0p001_nohh`.
  - `scripts/main_4hop_gr2st_ae0p05.sh` → job **25555510**, λ=0.05, prefix `full_4hop100_ae0p05_nohh`.
  - Both launched 2026-09-03, running (0.001 @ep0, 0.05 just started). 0.05 probes where over-sharpening starts to cost F1.
- Logs are **TensorBoard event files only** (no `metrics.csv`) — read scalars with `EventAccumulator`. Note
  `selections.csv` `epoch` runs one ahead of the TB `epoch` scalar (off-by-one index; same run). No code changed
  this session (only new sbatch scripts + docs).

### 2026-09-02 (later) — GR2ST attention-entropy regularization implemented _(details: `docs/20260902_gr2st_attention_entropy.md`)_
- Added a literal GR2ST attention-entropy auxiliary loss on the GATv2 post-softmax attention:
  `-mean_e[ sum_h alpha log(alpha+eps) ]`, added to the training objective as
  `λ·attention_entropy` (config `model.attention_entropy_weight`, default **0.0** = disabled/backward-compat;
  `model.attention_entropy_eps: 1.0e-9`). Requires `gnn="gat"` (raises `ValueError` otherwise).
- `GnnFs.forward` gained `return_attention_weights=False` (zero-weight path unchanged, returns just logits);
  `LitGnnFs.training_step` composes `cls + scGist_reg + λ·entropy` and returns `train_loss_total` (Lightning
  auto-optimization; no manual backward). Stacked GAT layers aggregated by **mean** (not sum). Entropy over
  all GAT edges in the (post-HalfHop) forward pass. Validation logs `val_attention_entropy` + sharpness
  diagnostics **as diagnostics only** — not in `val_loss_ce`, checkpoint still monitors `val_f1_overall`.
- `gfs/trainers/antelope.py` needed **no** loss-related change. New TB metrics: `train_loss_total`,
  `train_attention_entropy`, `train_loss_attention_entropy_weighted`, `train_attention_{top1_mass,
  effective_neighbors,normalized_entropy}` (+ `val_*` variants). `train_loss_ce` kept as backward-compat alias.
- Added `model.log_attention_stats` (default **false**) to **decouple measuring from regularizing**: set it
  true on a `weight=0` baseline to log the entropy + neighbourhood diagnostics without applying the loss, so
  baseline-vs-regularized attention stats are directly comparable. Numerically free (bit-identical logits);
  default path stays bit-identical to pre-change (verified). GAT-only (raises for gcn/sage).
- Tests: `tests/test_attention_entropy.py` (13, all pass on CPU) — backward-compat, entropy helper, gradients
  to GAT `att` params, multi-head/multi-layer, PyG per-dest normalization ≈1, non-GAT handling, training_step
  composition, tiny Lightning fits. `pytest` installed into the `gfs` env (not yet in `pyproject.toml` dev deps).
- **Not launched** — needs Slurm/GPU. Run baseline (`model.attention_entropy_weight=0.0`) vs `0.01` (GR2ST
  default) with everything else fixed; sweep `{0,1e-3,1e-2,1e-1}`. Success = sharper attention (entropy↓,
  top-1↑, eff-neighbours↓) with `val_f1_overall` held/improved — not merely "entropy decreased."

### 2026-09-02 (later) — GR2ST attention-entropy on/off comparison LAUNCHED _(details: `docs/20260902_gr2st_attention_entropy.md`)_
- Launched `scripts/main_4hop_gr2st_compare.sh` = Slurm array **25553166** (`--array=0-1`, both `rand_seed=0`):
  a paired 4-hop / 100ep / select-5 / HalfHop-OFF comparison where the ONLY difference is the attention loss.
  - task 0 (`25553166_0`, prefix `full_4hop100_ae0_nohh`): `attention_entropy_weight=0.0`, `log_attention_stats=true`.
  - task 1 (`25553166_1`, prefix `full_4hop100_ae0p01_nohh`): `attention_entropy_weight=0.01` (GR2ST default).
- Added the backward-compatible attention keys to `gfs/configs/antelope_4hop.yaml` (Hydra struct mode needed them
  present to override); baseline arm is otherwise identical to the seed-0 100ep 4-hop model (should track 0.6849 w).
- **On completion:** recompute test F1 from `test_pred.pt`; compare train/val `attention_entropy`, `top1_mass`,
  `effective_neighbors`, `normalized_entropy` between arms. If attention barely moves + F1 flat → raise λ to 0.03/0.1.

### 2026-09-02 (later) — 4-hop 300ep probe CANCELLED (wall-clock), to resume via checkpoint-resume _(details: `docs/4hop_2hop_comparison.md`)_
- Cancelled 4-hop 300ep seeds `25522893_0/_1` (were at ~ep68 / ~ep48 of 300). Measured cost is **~17 min/epoch
  → ~85h for 300ep**, over the 60h `--time` cap; the sbatch script's "~36h/seed" estimate was wrong. They'd have
  been killed at ~ep211/215, **mid-tau-anneal** (`"old"` schedule ties tau to `max_epochs`; tau≈0.078 at ep211,
  selection uncommitted) → partial checkpoints unusable. No results from this probe.
- **TO RESUME LATER (deferred):** chosen path is **checkpoint-resume** — requeue across multiple 60h blocks so the
  full 300ep anneal completes without a single >60h allocation (celltypes QOS likely caps `--time` <90h). Needs
  resume-from-checkpoint support added to `antelope_4hop.py` (`trainer.fit(ckpt_path=...)` + Slurm `--requeue`).
  Low priority: the 100ep result already answers 4hop-vs-2hop; this probe only tests longer tau exploration.

### 2026-09-02 — Job progress check + docs handoff _(details: `docs/4hop_2hop_comparison.md`, `docs/feature_ablation.md`)_
- Checked the three in-flight experiment groups from tfevents (`.out`/`.err` only show the model summary — the
  Lightning progress bar isn't flushed; read `lightning_logs/version_0/events*` and `selections.csv` instead).
- **300ep probe (`25522891`/`25522893`):** 2-hop seed 0 ✅ done — recomputed **0.7006 wtd / 0.5054 macro**
  (n=31,562), panel `{2,12,83,435,480}` (stayed on the {2,12,83,480} core). **Longer tau exploration did NOT
  help** — matches the 100ep 2-hop mean (0.7049). 2-hop seed 1 (~ep51), 4-hop seed 0 (~ep9) still running;
  4-hop seed 1 not yet started. Filled the probe progress table in `docs/4hop_2hop_comparison.md`.
- **feature-ablation (`25511099`):** panel 1 ✅ done — recomputed **0.6430 wtd / 0.5184 macro** (PERSIST panel
  `{2,25,83,207,225}`), **well below** the with-selection arm (0.7099 ± 0.0142 wtd, 25507593) → a frozen PERSIST
  panel does NOT match the learned selector so far. Panels 0/2/3/4 still training (~ep85–92/100). Filled partial
  results + run-status table in `docs/feature_ablation.md`; frozen-arm summary pending the four remaining panels.
- **baseline_comparisons:** the two comparison arms (`main_panel_persist_1x.sh`, `main_panel_graph_persist_shuffle.sh`)
  are **still not launched** (no log dirs) — only the GFS 4-hop reference (25507593) is done. No change to that doc.
- Docs synced: `docs/README.md` index statuses refreshed; `progress.md` running-jobs + Next steps updated.
  No code changes this session.

### 2026-08-22 — HalfHop 2x2 ablation complete (seed 0) _(details: `docs/20260820_deep_sample_shallow_loss.md`)_
- Jobs 25272602 (4hop/HHon), 25273001 (2hop/HHoff), 25273003 (4hop/HHoff) all finished cleanly (100 epochs,
  tested from best-val checkpoint). Grid in Current Status above; F1 recomputed from `test_pred.pt`.
- Best: 4-hop complete-field loss + HalfHop OFF = 0.7933 w / 0.7824 m (+0.014 / +0.021 over the 0.7789 baseline).
- HalfHop helps at 2-hop (+0.004) but hurts at 4-hop (−0.026) — complete-field win only appears with HalfHop off.
- Log dirs: `20260820_180945_abundant_4hop_seed0_s0`, `20260820_182036_abundant_select5_seed0_nohh_s0`,
  `20260820_182036_abundant_4hop_seed0_nohh_s0`.

### 2026-08-20 — Deep-sample / shallow-loss (4-hop) scaffolding _(details: `docs/20260820_deep_sample_shallow_loss.md`)_
- Fixes `trainmode=0`'s bug: it takes loss on all patch nodes, but 2-hop sampling only gives the *root* a
  complete 2-hop field. New `trainmode=2` samples `n_hops=4` and takes loss only on nodes within
  `loss_hops=2` of a seed (complete field), via `batch.loss_mask`. Message-passing stays 2 hops
  (`local_layers=2`, independent of sampling depth); val/test still root-only.
- Added `gfs/data/hemisphere_new_4hop.py`, `trainmode==2` in `antelope.py`, `antelope_4hop.py`,
  `antelope_abundant_4hop.yaml`, `scripts/main_abundant_4hop.sh`.
- Smoke-tested: bs=16 → ~3.7k nodes/patch; **bs=64 → ~14k nodes, ~4,650 complete-field loss nodes/batch,
  ~1,166 steps/epoch** — matched to the abundant baseline (loss-node volume ~4,650 vs ~4,147, steps ~1,166 vs 1,167).
  Forward+backward + selection-gradient verified. Config set to bs=64 for the direct comparison. No full run launched yet.

### 2026-08-16
- Docs/tracker housekeeping only (no code/research changes): made `CLAUDE.md` concise (durable guidance
  only), moved architecture internals + volatile conventions to this file's Reference appendix, reorganized
  `progress.md` into Current Status + Development Log, added `docs/README.md` index and a doc-naming convention,
  renamed `docs/0811_exp.md` → `docs/20260811_abundant_subclass.md` to fit the convention.
- Added the intact root-loss (`trainmode=1`) panel `{5,83,207,225,480}` (job 24944818) to
  `does-the-graph-help.md` Evidence 2c, alongside the shuffle root-loss `{2,38,225,435,480}` (24950004),
  with test metrics from the `.out` files: shuffle-root 0.6037 / intact-root 0.6100 weighted F1 (both s0).
  Both starved arms drift off `{2,12,83,435,480}` and score ≈0.02 below the full-exposure shuffle (0.6261).

### 2026-08-11 — Abundant seed-0 arms complete _(details: `docs/20260811_abundant_subclass.md`)_
- Runs: default → `data/logs/20260810_204705_abundant_select5_seed0_s0/`; shuffle → `..._shuffle_s0/`.
- Panels **identical** `{2, 12, 38, 83, 480}` (indices; Ensembl IDs in the doc) → selection is graph-invariant here.
- Test (recomputed from `test_pred.pt`, n=31,563): weighted F1 **0.7789** (intact) vs 0.7365 (shuffle), +0.042;
  macro 0.7618 vs 0.7106, +0.051. Graph gain concentrated on 004 L6 IT Glut (+0.155 F1).

### 2026-08-10 — Abundant-subclass experiment scaffolding
- Data verified `data/0810_abudant_classes_only_top6/`: 7 subclasses (top-6 + `other`), 500 genes;
  train 74,634 / val 31,563 / test 31,563; has `spatial_connectivities`, xyz.
- Added config `gfs/configs/antelope_abundant.yaml` (`n_labels: 7`) + scripts `main_abundant.sh` /
  `main_abundant_shuffle.sh`. Smoke-tested both loaders (74634×503, 7 classes, 1,567,314 edges); confirmed
  shuffle keeps edges+xyz identical and permutes the gene block. Created `CLAUDE.md` and `progress.md`.

### Prior work (pre-2026-08-10)
- Base GFS pipeline (model, data modules, trainers, configs) pre-existing and working.
- Shuffle-cell control methodology and prior baseline/panel findings established — see
  `docs/does-the-graph-help.md` and `MEMORY.md`.

---

# Reference (moved from CLAUDE.md — durable but subject to change)

## Architecture / key concepts
- **Model** `gfs/models/antelope.py`: `GnnFs` (backbone) + `LitGnnFs` (Lightning wrapper).
  - Feature selection via `fs_method`: `persist` (default; concrete/Gumbel-softmax k-hot mask over genes,
    `logits` shape `(n_select, gene_ch)`), `scGist` (L1-regularized), or `none` (frozen panel, no selection).
  - Input `x` = `[gene_exp (n_genes) | xyz (3)]`; `gene_exp_ind` / `xyz_ind` slice them back out.
  - Selection mask is soft (Gumbel-softmax, temperature `tau` annealed by `tau_schedule`, `tautype="old"` is
    the standard) during training, hard (argmax) at val/test.
  - Extras: HalfHop augmentation, xyz residual, x residual (MLP), jumping-knowledge, per-subgraph xyz
    mean-centring (vectorized — see memory `vectorized-xyz-centring-perf-fix`).
  - Selected gene indices → `<logdir>/selections.csv` each epoch; test predictions+labels → `<logdir>/test_pred.pt`.
- **Data modules** `gfs/data/hemisphere_new*.py`: build a PyG graph from AnnData. `NeighborLoader` samples
  n-hop patches (`n_hops=2`); `NeighborLoaderMod` tags per-patch `subgraph_id`. Val/test masks come from `adata.obs`.
- **Trainers** `gfs/trainers/antelope*.py`: thin Hydra+Lightning entrypoints. Each variant just swaps which
  data module it imports; all read `gfs/configs/antelope*.yaml`. `val_f1_overall` is the checkpoint monitor.

### Data-module variants (the import at top of each trainer selects behavior)
- `hemisphere_new.py` — standard: cells aligned to their real spatial graph nodes.
- `hemisphere_new_shuffle.py` — **shuffled-cell control**: gene expr + labels permuted TOGETHER by one shared
  reproducible permutation; xyz + edges left intact. Ablates neighborhood signal, preserves expr↔label pairing.
- `hemisphere_new_shuffle_within.py` — shuffles within each val/test split only (avoids re-partitioning splits).
- `hemisphere_new_shuffle_epoch.py` — re-draws permutation each epoch (empirically same as fixed shuffle).
- `hemisphere_new_panel.py` / `panel_tabular.py` — frozen-panel (`fs_method=none`) baselines.
- `hemisphere_new_4hop.py` — **complete-receptive-field training**: train samples `n_hops` (=4), attaches
  `batch.loss_mask` = nodes within `loss_hops` (=2, GNN depth) of a seed; used by `trainmode=2`. Val/test
  sample `loss_hops` hops, root-only eval. Requires `data.loss_hops` in config + `antelope_4hop.py` trainer.
  See `docs/20260820_deep_sample_shallow_loss.md`.

### trainmode (LitGnnFs.training_step)
`model.trainmode` selects which nodes contribute to the training loss: **0** = all patch nodes (default;
most patch nodes have incomplete receptive fields under 2-hop sampling), **1** = root/seed nodes only,
**2** = nodes within `loss_hops` of a seed (all with complete fields; needs `hemisphere_new_4hop`'s
`loss_mask`). Val/test/predict always score root nodes only, regardless of trainmode.

### Eval-time HalfHop (load-bearing gotcha — affects ALL experiments)
`HalfHop.forward` has **no `self.training` guard**, and `LitGnnFs` calls `self.transform(batch)` in
val/test/predict too. So with `model.halfhop=true` (the default in every abundant/shuffle config), eval
message-passes over a HalfHop-perturbed graph (slow nodes inserted on ~p=0.5 of edges), NOT the raw
2-hop graph. Every reported F1 with halfhop on (incl. the 0.7789 baseline) carries this. To eval on the
raw graph, set `model.halfhop=false` (off for train AND eval) — see the no-HH ablation arms. Left ON in
the primary comparison so old vs new stay apples-to-apples; not yet gated to train-only (a possible future change).

## Conventions / config (subject to change)
- **New experiment = new config + new sbatch script**, reusing an existing trainer where possible. Change the
  data module only by which trainer you point at (or add a new one-line-different trainer).
- **`data.prefix`** names the run (feeds `logs/<datetime>_<prefix>_s<seed>/` and checkpoint dirs). Keep it descriptive.
- Set **`n_labels`** to match the dataset's subclass count (base=22; abundant-top6 subset=7) — it flows to `model.out_ch`.
- `gfs/configs/antelope.yaml` = base config (full-data: 500 genes, 22 subclasses). Region-restricted /
  experimental subsets live in dated `data/` subdirs, e.g. `data/0810_abudant_classes_only_top6/` (7 subclasses).
- `scripts/main*.sh` run on A100, `-p celltypes`. `outs/` holds `out-comms-<jobid>.out` / `error-comms-<jobid>.err`.
- Metric of record: `test_f1_overall` (weighted); also track macro. Recompute from `test_pred.pt`, not plots.
- **Significance convention (paired 4-hop-vs-2-hop, n=5 seeds):** report the **paired t-test as the primary
  test**, with **Cohen dz + per-seed sign count + a cell-level paired bootstrap** as support. **Wilcoxon is
  deprecated for the significance call** — n=5 pins its minimum two-sided p at 0.0625, so it can never clear
  α=0.05 even on a clean 5/5 sweep (uninformative floor).
  **Bootstrap (redone 2026-09-25) = cell-level paired**, `gfs/analysis/paired_cell_bootstrap.py`, B=10,000:
  one test-cell resample (with replacement) shared across all 5 seeds + both arms, per-seed F1 differenced
  within seed then averaged; 95% percentile CI + 2-sided p. This RETIRED the earlier seed-level paired
  bootstrap (B=50k). Rationale: all cells are from ONE biological sample with a fixed split, so the only
  resamplable unit is the cell (seed pairing/averaging preserved). **Scope: within-sample cell uncertainty,
  NOT across-sample/donor generalization** — it is far higher-powered than a seed-level test (clears α=0.05
  on ~20/22 grouping rows) and routinely disagrees with the t-test, so treat it as direction-corroborating
  only, never the generalization headline. Consolidated results:
  `docs/20260925_4hop_2hop_cell_bootstrap_summary.md`.
  Pairing is valid because 2-hop/4-hop seed_i score the identical test set (labels element-wise equal); note the
  split is fixed across seeds, so the n=5 spread is training-run stochasticity, not data/split uncertainty.
  Applied across all 7 4hop-vs-2hop docs 2026-09-25; the 09-07 & 09-14 grouping docs already used t as primary
  (their Wilcoxon columns are retained as pedagogical "maximally-in-favor-but-floored" context).
- The shuffle control ablates the GRAPH used for *selection*; it is NOT a claim about the model. At matched
  panel size the intact graph is worth real F1 (memory `panel-graph-arm-noise-robustness`).

(Active bugs/blockers are tracked in "Known blockers / issues" above.)
