# docs/ — index

Focused technical write-ups: experiment runs, analyses, decisions, investigations. One topic per file.
Prefer updating an existing doc over spawning an overlapping one. `progress.md` and `CLAUDE.md` link here
rather than absorbing this detail.

## Naming convention
- **Dated experiment run** → `YYYYMMDD_topic.md` (e.g. `20260811_abundant_subclass.md`). Ties the doc to a specific set of runs.
- **Durable investigation / analysis** (spans many runs, no single date) → topic name (e.g. `does-the-graph-help.md`).

## Contents
- **`conclusion.md`** — **top-level summary**: the two headline claims (the spatial graph helps
  *classification* but not *gene selection*; the learned 4-hop panel is the best panel), each decomposed
  into the specific `>` comparisons with their numbers and links to the underlying docs. Start here. Linked
  from the repo `README.md`.
- **`20260925_4hop_2hop_cell_bootstrap_summary.md`** — **consolidated** 4-hop-vs-2-hop results across ALL
  label groupings (full22, abundant top-6, next-6, diverse6 bal/maxdisp, deep_sample Glut/NN/Gaba,
  finegrained gaba_fine/cge/mge), all re-tested with one uniform **cell-level paired bootstrap** (B=10,000,
  shared cell resample across 5 seeds, `gfs/analysis/paired_cell_bootstrap.py`). Single authoritative table
  (Δ, boot CI/p, paired-t p, primary call). Key read: bootstrap sig almost everywhere (within-sample power),
  t-test sig only for abundant top-6, maxdisp-weighted, Glut-weighted. Supersedes the per-doc seed-level
  bootstrap (B=50,000).
- **`20260919_diverse6_maxdisp.md`** — "diverse-6" (max-dispersion) abundant-style experiment: of the top-12
  by abundance, the 6 whose mean expression profiles are **maximally spread** (max total pairwise distance;
  correlation & Euclidean agree): 007 L2/3 IT Glut, 052 Pvalb / 053 Sst Gaba, 319 Astro / 327 Oligo / 333 Endo
  NN (1 Glut / 2 Gaba / 3 NN) + `other` (7 lbl). Distinct from top-6 (3/6) and next-6 (3/6). Same deep-sample
  protocol. **DONE 2026-09-24 (array `26443773`)** — 4-hop wins: weighted +0.0185 (5/5 seeds, t p=0.016),
  macro +0.0226 (5/5, ns); strongest abundant-style 4-hop signal yet. See Results section.
- **`20260919_diverse6_balanced.md`** — "diverse-6" (family-balanced 2/2/2) companion: of the top-12 by
  abundance, a family-balanced diverse set via max-min centroid dispersion — 007 L2/3 IT / 030 L6 CT Glut,
  052 Pvalb / 053 Sst Gaba, 319 Astro / 327 Oligo NN + `other` (7 lbl). Distinct from top-6 (3/6) and next-6
  (3/6); differs from max-dispersion by one subclass (030 L6 CT ↔ 333 Endo). **Scaffolding COMPLETE,
  smoke-tested, NOT yet launched** — configs `antelope_diverse6_bal{,_4hop}.yaml`, launcher
  `scripts/main_diverse6_bal_all.sh`.
- **`20260919_abundant_next6.md`** — "next-6" abundant-style experiment: keep the **next 6** most-abundant
  subclasses by cell count (ranks 7–12: 327 Oligo NN, 005 L5 IT / 022 L5 ET / 032 L5 NP Glut, 052 Pvalb /
  053 Sst Gaba) + `other` (7 labels), sibling of the top-6 abundant set. Built from the default 22-label full
  data via `build_subclass_variant.py` (same source/split as top-6). First abundant-style set spanning all
  three families (3 Glut / 2 Gaba / 1 NN). Same deep-sample protocol (4-hop/tm2 vs 2-hop/tm0, HalfHop-OFF,
  select-5, seeds 0–4). **DONE 2026-09-24 (array `26443761`)** — positive 4-hop trend, ns: weighted +0.0082
  (4/5), macro +0.0302 (4/5, dz≈0.99, p=0.09); near ceiling. Matches original top-6. See Results section.
- **`20260907_subclass_grouping_deep_sample.md`** — deep_sample (4-hop/tm2 vs 2-hop/tm0, HalfHop-OFF,
  n_select=5, seeds 0–4) rerun on three subclass-grouping variants (Glut/NN/Gaba, each = kept family +
  `other`, built from the fine-stratified full data via `gfs/data/build_subclass_variant.py`). **COMPLETE**
  (array `25585854`, 30 runs): all 6 dataset×metric Δ positive but only Glut-weighted significant
  (+0.0098, p=0.0097); NN ns, Gaba trends up (macro 5/5, p=0.066). Cross-family verdict + panels in the doc.
- **`20260914_subclass_grouping_finegrained.md`** — Same experiment (4-hop vs 2-hop, select-5, HalfHop-off,
  seeds 0–4) on the enriched label set (`data/0914_all_subclasses/`, 34 labels, less collapsed into `other`).
  **Gaba arm COMPLETE + ANALYZED 2026-09-20** (Slurm array `26380063`, 30/30 runs): three keep-sets —
  **gaba_fine** (9 lbl, all 8 Gaba subclasses), **gaba_cge** (4 lbl, Vip/Sncg/Lamp5), **gaba_mge** (6 lbl,
  Pvalb/Pvalb-chandelier/Sst/Sst-Chodl/Lamp5-Lhx6) — built via `build_subclass_variant.py --src 0914_all_subclasses`.
  **Verdict: nothing significant at n=5** — weighted at ceiling (0.98–0.99, all Δ positive but ns); macro fine/cge
  tie, **mge +0.054 (ns), driven by rare laminar types**; CGE-vs-MGE points the predicted way (MGE>CGE) but is a
  trend. Rare-class caveat confirmed. Results + panels in the doc. Fuller Glut-16 / NN-11 / full-34 + type-2
  partition groupings still PLANNING (not built).
  (menu + rare-class caveat in the doc).
- **`20260902_gr2st_attention_entropy.md`** — GR2ST-style GATv2 attention-entropy regularization: a
  literal auxiliary entropy loss on post-softmax GAT attention (config `model.attention_entropy_weight`,
  default `0.0`), plus a `log_attention_stats` flag to log entropy/neighbourhood diagnostics even when the
  loss is off. Integrated into `LitGnnFs.training_step`. Implemented + unit-tested
  (`tests/test_attention_entropy.py`, 15 pass). **λ sweep RUNNING** (seed 0, 4-hop/100ep/select-5/HalfHop-OFF,
  identical splits): λ=0.0 `25553166_0` + λ=0.01 `25553166_1` (`main_4hop_gr2st_compare.sh`), λ=0.001 `25555463`
  (`main_4hop_gr2st_ae0p001.sh`), λ=0.05 `25555510` (`main_4hop_gr2st_ae0p05.sh`); analysis pending completion.
- **`baseline_comparisons.md`** — Test metrics for the regenerated PERSIST select-5 panels via a **no-graph
  (self-edges)** and a **shuffled-graph** `LitGnnFs`, compared to the GFS **full-graph 4-hop** model
  (job 25507593; reference ✅ **complete** 2026-09-01: 0.7099 ± 0.0142 wtd / 0.5272 ± 0.0141 macro). Three-arm
  decomposition at a frozen panel: panel-only floor → +exposure/noisy conv (shuffled) → +real neighbours
  (intact/GFS). `rand_seed=0` all runs. Datasets built (`exposure/persist_1x_seed{0-4}`), both arms
  smoke-tested; **the self-edges + shuffled-graph arms are NOT yet launched** (pending user `sbatch`).
- **`feature_ablation.md`** — Does the learned selector beat a frozen PERSIST select-5 panel, holding the
  4-hop backbone fixed? Full 22-label data, `fs_method=none` (frozen panel) vs `persist` (selection),
  seeds 0–4. With-selection arm ✅ done (Slurm 25507593). Frozen arm (Slurm 25511099) ✅ **complete** — all
  5 panels: **0.6847 ± 0.0254 wtd / 0.5198 ± 0.0084 macro** vs selector 0.7099 ± 0.0142 / 0.5272 ± 0.0141
  (selector +0.025 wtd / +0.007 macro, within ~1 sd; frozen deficit is variance not ceiling).
- **`feature_ablation_abundant.md`** — abundant 7-label counterpart of `feature_ablation.md`. With-selection
  reference reused from `20260820_deep_sample_shallow_loss.md` (0.7851 ± 0.0073 wtd / 0.7764 ± 0.0067 macro).
  Frozen arm scaffolded (config `antelope_abundant_4hop_panel.yaml`, `main_abundant_4hop_panel_seeds.sh`) —
  **pending: generate abundant PERSIST panels, then launch** (user `sbatch`).
- **`4hop_2hop_comparison.md`** — 4-hop (complete-field, `trainmode=2`) vs 2-hop baseline (`trainmode=0`)
  on the **full 22-label** data, GFS select-5, HalfHop off, seeds 0–4. ✅ **complete** — 4-hop does NOT beat
  2-hop (tied: 0.7099 vs 0.7049 wtd; abundant win fails to replicate). Plus a 300-epoch exploration probe:
  2-hop (job 25522891) both seeds done (0.7006/0.7114, no gain from longer exploration); **4-hop (job 25522893)
  CANCELLED** for wall-clock (~85h > 60h cap) — to resume later via checkpoint-resume.
- **`20260820_deep_sample_shallow_loss.md`** — 4-hop sample / 2-hop loss training: fixes `trainmode=0`'s
  incomplete-neighborhood loss nodes by sampling `n_hops=4` and taking loss only on nodes with a complete
  2-hop field (`trainmode=2`, `batch.loss_mask`). Scaffolding + smoke test; no full run yet.
- **`20260811_abundant_subclass.md`** — Abundant-subclass experiment (7 labels), seed 0: intact graph vs shuffled-cell control at
  `n_select=5`. Result: identical panel `{2,12,38,83,480}`; intact graph gives +0.042 weighted F1 on classification.
- **`does-the-graph-help.md`** — The resolved graph question: **yes** for classification (+0.082 weighted F1),
  **no** for gene selection. Two separate questions with opposite answers; the split is the whole point.
- **`old-claude.md`** — Archived long-form research log (read-only history). Deep background predating the
  focused docs above.
