# 2026-09-19 — "diverse-6" abundant-style experiment (max-dispersion)

**Question.** The abundant experiments so far select subclasses by **cell count**: top-6
(`docs/20260811_abundant_subclass.md`, `docs/20260820_deep_sample_shallow_loss.md`) and next-6
(`docs/20260919_abundant_next6.md`). This arm instead asks: of the **top-12 by abundance**, which **6 are the
most transcriptomically diverse**, and does the deep-sample (4-hop vs 2-hop) pattern look different on a
maximally-spread label set than on the abundance-ordered ones?

## Selection — max-dispersion of expression centroids

For each of the top-12-by-abundance subclasses, compute its mean expression vector (centroid) over the 500
genes on the training data, then choose the 6-subset that **maximizes total pairwise distance** between
centroids. Robust across metrics — **correlation distance and Euclidean-on-z-scored genes give the identical
set** under the max-sum criterion (brute force over all C(12,6)=924 subsets).

**Selected 6 (family mix 1 Glut / 2 Gaba / 3 NN):**

| subclass | family | abundance rank | valtest count |
|---|---|---|---|
| 007 L2/3 IT CTX Glut | Glut | 4 | 6126 |
| 052 Pvalb Gaba | Gaba | 10 | 2119 |
| 053 Sst Gaba | Gaba | 11 | 1538 |
| 319 Astro-TE NN | NN | 3 | 6981 |
| 327 Oligo NN | NN | 7 | 3330 |
| 333 Endo NN | NN | 5 | 5043 |

Interpretation: max-dispersion grabs **all 3 non-neuronal types** (glia + endothelial — far from neurons and
from each other), **both GABAergic interneuron types**, and **one excitatory neuron** — maximal spread across
the taxonomy rather than several cortical-Glut layers. Category order (sorted kept, then `other`; codes 0→6):
`007 L2/3 IT` · `052 Pvalb` · `053 Sst` · `319 Astro-TE NN` · `327 Oligo NN` · `333 Endo NN` · `other`.

## Distinctness from the other abundant-style sets

All four sets are distinct (none identical):

| pair | overlap /6 |
|---|---|
| max-dispersion vs top-6 | 3 (007, 319, 333) |
| max-dispersion vs next-6 | 3 (052, 053, 327) |
| max-dispersion vs family-balanced (`docs/20260919_diverse6_balanced.md`) | 5 (differ only: this has 333 Endo; balanced has 030 L6 CT) |

So max-dispersion is a 3-from-top6 / 3-from-next6 hybrid — genuinely different from both submitted abundance sets.

## Dataset

Built from the **default 22-label full data** via `gfs/data/build_subclass_variant.py` (same source/split as
top-6 and next-6, so all comparable):

```bash
python -m gfs.data.build_subclass_variant \
  --keep "319 Astro-TE NN,333 Endo NN,327 Oligo NN,052 Pvalb Gaba,053 Sst Gaba,007 L2/3 IT CTX Glut" \
  --out 20260919_diverse6_maxdisp
```

Output: `data/20260919_diverse6_maxdisp/isocortex_{train,valtest}_final.h5ad`, 500 genes, **7 labels**.
Build verification: train 74,633 / valtest 63,124; per-class val/test balance ~0.5000, **max |val−test| = 1**;
`other` = 37,987 valtest.

## Protocol

Byte-identical to the abundant/next-6 deep-sample protocol (only path/name differ):
- **2-hop** — `antelope.py`, `n_hops=2`, `trainmode=0`, config `antelope_diverse6_maxdisp.yaml`.
- **4-hop** — `antelope_4hop.py`, `n_hops=4`, `loss_hops=2`, `trainmode=2`, config `antelope_diverse6_maxdisp_4hop.yaml`.
- Both: `n_select=5`, `halfhop=false`, 100 epochs, `lr=1e-3` const, best-`val_f1_overall` ckpt, seeds 0–4.
- Launcher `scripts/main_diverse6_maxdisp_all.sh` — array `0-9%4`; task → `(arm=id/5, seed=id%5)`.

## Status — DONE 2026-09-24

**COMPLETE — Slurm array `26443773`, all 10/10 tasks finished** (2 arms × 5 seeds, 100 ep + `test_pred.pt`).
Log dirs `data/logs/*_diverse6_maxdisp_*_nohh_s*/`.

## Results — 2026-09-24

F1 recomputed from `test_pred.pt` (argmax of logits vs labels; population sd, ddof=0), paired by seed.
⚠️ n=5 → very low power; read the paired t-test / bootstrap / Cohen dz + sign count together, not any one p.

| metric | 2-hop | 4-hop | Δ (4h−2h) | sign | t p | bootstrap 2-sided p | Cohen dz |
|---|---|---|---|---|---|---|---|
| weighted-F1 | 0.8192 ± 0.0146 | 0.8378 ± 0.0092 | **+0.0185** | **5/5** | **0.016** | **<0.0002** | +1.79 |
| macro-F1 | 0.7285 ± 0.0090 | 0.7510 ± 0.0260 | +0.0226 | 5/5 | 0.106 | **<0.0002** | +0.93 |

Bootstrap redone 2026-09-25 = **cell-level paired bootstrap**, B=10 000 (`gfs/analysis/paired_cell_bootstrap.py`):
one draw of test-cell indices with replacement is applied to **all 5 seeds × both arms**, per-seed F1 is
recomputed, differenced within seed, then averaged across the 5 seeds; 95% percentile CI + 2-sided p.
Weighted 95% CI [+0.0167, +0.0204], macro 95% CI [+0.0196, +0.0256]; both had 0/10 000 replicates cross zero
(p<0.0002). So the cell-level bootstrap now flags **both** metrics significant on this sample — including macro,
which the primary paired t-test still calls ns (p=0.106). That is a **power/scope difference, not a
contradiction**: ⚠️ the cell-level bootstrap measures cell-level uncertainty **within one biological sample**
(31 562 test cells) and is far higher-powered than a 5-seed paired test — it does **NOT** establish across-sample/
donor generalization. The **paired t-test remains the primary test** for the generalization claim (weighted a
real win, macro a trend). (Supersedes the earlier seed-level paired bootstrap, B=50 000.)

Pairing verified: each seed's 2-hop and 4-hop runs score the **identical test set in identical order**
(`test_pred.pt` labels element-wise equal, n=31 562), so Δ(seed) is a matched diff. Replicate-unit caveat:
the split is **fixed across seeds** (all 5 share the same test-label vector) — the seed randomizes only model
init / sampling / concrete-selection, not the data partition. The n=5 spread captures training-run
stochasticity on one fixed split, not split/sampling uncertainty.

Per-seed (weighted / macro):

| seed | 2-hop | 4-hop | Δw | Δm |
|---|---|---|---|---|
| 0 | 0.7992 / 0.7225 | 0.8290 / 0.7273 | +0.0299 | +0.0048 |
| 1 | 0.8419 / 0.7429 | 0.8525 / 0.7947 | +0.0106 | +0.0518 |
| 2 | 0.8228 / 0.7231 | 0.8284 / 0.7253 | +0.0056 | +0.0022 |
| 3 | 0.8241 / 0.7188 | 0.8439 / 0.7647 | +0.0197 | +0.0459 |
| 4 | 0.8083 / 0.7350 | 0.8352 / 0.7430 | +0.0269 | +0.0081 |

**Verdict: the strongest 4-hop signal in the abundant-style family so far.** Weighted improves on **every seed**
(t p=0.016, large effect dz≈1.8) — a genuine, significant win. Macro also positive on all 5 seeds (+0.0226) but
not t-significant: the gain concentrates in seeds 1 (+0.052) and 3 (+0.046) while 0/2/4 barely move, inflating
4-hop macro variance. Consistent with the cross-family pattern (`docs/20260907_subclass_grouping_deep_sample.md`)
— deep sampling helps the harder separations more than the ceiling-bound weighted metric, but here even weighted
clears significance because this label set sits well below ceiling (~0.82 vs next-6's ~0.92).

**Panels (final-epoch `selections.csv` → gene_symbol, freq x/5 seeds):**
- 2-hop: Grik3:5, Kcnk9:5, Zfp536:5, Prom1:3, Slc7a10:3, (Adgrf5/Cldn5/Grm1/Lpl):1
- 4-hop: Grik3:5, Slc7a10:4, Zfp536:4, Kcnk9:3, Prom1:2, Ptprm:2, (others):1

Core panel is stable across arms (Grik3, Zfp536, Kcnk9, Slc7a10); 4-hop leans a little more on Slc7a10/Ptprm.

Recompute: `gfs/analysis/gaba_grouping_f1.py` helpers (`find_run`/`f1s`/`final_panel`/`paired_stats`) reused with
stem `diverse6_maxdisp`, symbols from `data/20260919_diverse6_maxdisp/isocortex_train_final.h5ad`.
