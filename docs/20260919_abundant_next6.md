# 2026-09-19 — "next-6" abundant-style experiment (ranks 7–12 by cell count)

**Question.** The original abundant-subclass experiment (`20260811_abundant_subclass.md`,
`20260820_deep_sample_shallow_loss.md`) kept the **top-6 most-abundant subclasses** (by cell count) + `other`.
This is the direct sibling one window down: keep the **next 6** most-abundant subclasses (ranks 7–12) + `other`,
and run the same protocol. Does the deep-sample (4-hop) vs 2-hop pattern, and the graph's contribution, look the
same on a less-abundant / more-mixed-family label set?

## Dataset

Built from the **default 22-label full data** (`data/isocortex_{train,valtest}_final.h5ad`) via
`gfs/data/build_subclass_variant.py` — the same source and (fine-stratified) val/test split as the original
top-6 set, so the two are directly comparable. Command:

```bash
python -m gfs.data.build_subclass_variant \
  --keep "327 Oligo NN,005 L5 IT CTX Glut,022 L5 ET CTX Glut,052 Pvalb Gaba,053 Sst Gaba,032 L5 NP CTX Glut" \
  --out 20260919_abundant_next6
```

Output: `data/20260919_abundant_next6/isocortex_{train,valtest}_final.h5ad`, 500 genes, **7 labels**
(6 kept + `other`). Everything except `obs['subclass']` copied verbatim; all cells kept, no cell changes split.

**Selection recipe = rank subclasses by cell count (`obs['subclass'].value_counts()`), take ranks 7–12.**
NOT from `subclass_edge_pair_counts.csv` (that is a pairwise spatial-edge table, no single-subclass ranking).
The `subclass_edge_pair_counts.csv` played no role in either the original top-6 or this next-6.

Ranking (valtest counts), with the original top-6 for context:

| rank | subclass | family | valtest count | set |
|---|---|---|---|---|
| 1 | 006 L4/5 IT CTX Glut | Glut | 11387 | original top-6 |
| 2 | 030 L6 CT CTX Glut | Glut | 7637 | original top-6 |
| 3 | 319 Astro-TE NN | NN | 6981 | original top-6 |
| 4 | 007 L2/3 IT CTX Glut | Glut | 6126 | original top-6 |
| 5 | 333 Endo NN | NN | 5043 | original top-6 |
| 6 | 004 L6 IT CTX Glut | Glut | 4117 | original top-6 |
| **7** | **327 Oligo NN** | **NN** | **3330** | **next-6** |
| **8** | **005 L5 IT CTX Glut** | **Glut** | **2993** | **next-6** |
| **9** | **022 L5 ET CTX Glut** | **Glut** | **2180** | **next-6** |
| **10** | **052 Pvalb Gaba** | **Gaba** | **2119** | **next-6** |
| **11** | **053 Sst Gaba** | **Gaba** | **1538** | **next-6** |
| **12** | **032 L5 NP CTX Glut** | **Glut** | **1048** | **next-6** |

Category order (sorted kept, then `other`; `.cat.codes` 0→6):
`005 L5 IT` · `022 L5 ET` · `032 L5 NP` · `052 Pvalb` · `053 Sst` · `327 Oligo NN` · `other`.

Family mix vs the original: original top-6 = **4 Glut + 2 NN**; next-6 = **3 Glut + 2 Gaba + 1 NN** — the
next-6 is the first abundant-style set to include GABAergic interneurons, so it exercises all three families.

Build verification (from the build script's report): train 74,633 / valtest 63,124 cells; per-class val/test
balance ~0.5000 with **max |val−test| = 1** (inherits the fine-stratified split); `other` = 49,916 valtest
(much larger than the top-6's `other` = 21,835, since more abundant classes now fall into it).

## Protocol

Byte-identical to the abundant deep-sample / subclass-grouping protocol (only path / name differ):

- **2-hop baseline** — `antelope.py`, `n_hops=2`, `trainmode=0`, config `antelope_next6.yaml`.
- **4-hop complete-field** — `antelope_4hop.py`, `n_hops=4`, `loss_hops=2`, `trainmode=2`, config
  `antelope_next6_4hop.yaml`.
- Both: `n_select=5`, `halfhop=false` (train + eval, raw-graph eval), 100 epochs, `lr=1e-3` constant,
  best-`val_f1_overall` checkpoint, seeds 0–4.
- Launcher `scripts/main_next6_all.sh` — one throttled array `--array=0-9%4`; task id → `(arm=id/5, seed=id%5)`,
  arm 0 = 2-hop, arm 1 = 4-hop.

## Status — DONE 2026-09-24

**COMPLETE — Slurm array `26443761`, all 10/10 tasks finished** (2 arms × 5 seeds, 100 ep + `test_pred.pt`).
Log dirs `data/logs/*_next6_*_nohh_s*/`.

## Results — 2026-09-24

F1 recomputed from `test_pred.pt` (argmax vs labels; population sd, ddof=0), paired by seed.
⚠️ n=5 → very low power; the paired t-test is primary; Cohen dz + sign count support it. The bootstrap here is
the cell-level paired bootstrap (within-sample, not generalization — see method note below).

| metric | 2-hop | 4-hop | Δ (4h−2h) | sign | t p | bootstrap 2-sided p | Cohen dz |
|---|---|---|---|---|---|---|---|
| weighted-F1 | 0.9216 ± 0.0072 | 0.9298 ± 0.0061 | +0.0082 | 4/5 | 0.151 | <0.0002 (CI [+0.0071,+0.0093]) | +0.79 |
| macro-F1 | 0.7788 ± 0.0317 | 0.8090 ± 0.0269 | +0.0302 | 4/5 | 0.092 | <0.0002 (CI [+0.0265,+0.0340]) | +0.99 |

Bootstrap redone 2026-09-25 = **cell-level paired bootstrap**, B=10 000 (`gfs/analysis/paired_cell_bootstrap.py`):
shared test-cell resample across all 5 seeds × both arms, per-seed F1 differenced within seed then averaged;
95% percentile CI + 2-sided p. Weighted CI [+0.0071,+0.0093], macro CI [+0.0265,+0.0340], both 0/10 000 crossed
zero (p<0.0002). ⚠️ Scope: cell-level uncertainty within one biological sample — NOT across-sample/donor
generalization; far higher-powered than the seed-level paired t-test and can disagree with it. Supersedes the
earlier seed-level bootstrap (B=50 000).

Pairing verified: each seed's 2-hop and 4-hop runs score the **identical test set in identical order**
(`test_pred.pt` labels element-wise equal, n=31 562), so Δ(seed) is a matched diff. Replicate-unit caveat:
the split is **fixed across seeds** (all 5 share the same test-label vector) — the seed randomizes only model
init / sampling / concrete-selection, not the data partition. The n=5 spread captures training-run
stochasticity on one fixed split, not split/sampling uncertainty.

Per-seed (weighted / macro):

| seed | 2-hop | 4-hop | Δw | Δm |
|---|---|---|---|---|
| 0 | 0.9171 / 0.7617 | 0.9194 / 0.7700 | +0.0023 | +0.0083 |
| 1 | 0.9358 / 0.8403 | 0.9290 / 0.8315 | −0.0068 | −0.0088 |
| 2 | 0.9197 / 0.7778 | 0.9386 / 0.8457 | +0.0189 | +0.0679 |
| 3 | 0.9173 / 0.7558 | 0.9305 / 0.8030 | +0.0132 | +0.0472 |
| 4 | 0.9179 / 0.7583 | 0.9314 / 0.7948 | +0.0135 | +0.0365 |

**Verdict: positive 4-hop trend on both metrics, not significant by the primary (paired t) test.** Macro +0.0302
(dz≈0.99, medium-large) and weighted +0.0082, both 4/5 seeds favorable, but seed1 regresses (Δw −0.007, Δm −0.009)
and weighted is near ceiling (~0.92), so neither clears α=0.05 on the paired t-test — which stays primary. The
higher-powered cell-level bootstrap now flags **both** weighted and macro as significant on this sample (p<0.0002,
CIs clear of zero), but that is a within-sample statement (cell-level uncertainty on one fixed split), not
across-sample generalization, so it does not overturn the t-test call. Same direction and rough magnitude as the original top-6 abundant
result (`20260820_deep_sample_shallow_loss.md`: +0.0132 wtd / +0.0181 macro). Weaker than diverse6-maxdisp,
where the label set sits below ceiling and weighted reaches significance (`docs/20260919_diverse6_maxdisp.md`).

**Panels (final-epoch `selections.csv` → gene_symbol, freq x/5 seeds):**
- 2-hop: Nxph1:5, Parm1:5, Grik3:3, Sox10:3, Sulf2:3, Zfp536:3, (Mog/Ptprm/Tox):1
- 4-hop: Nxph1:5, Parm1:5, Grik3:4, Sox10:3, Ptprm:2, Sulf2:2, Zfp536:2, (Kcnk9/Mog):1

Core panel shared (Nxph1, Parm1, Grik3, Sox10); 4-hop uses Sulf2/Zfp536 slightly less.

Recompute: `gfs/analysis/gaba_grouping_f1.py` helpers reused with stem `next6`, symbols from
`data/20260919_abundant_next6/isocortex_train_final.h5ad`.
