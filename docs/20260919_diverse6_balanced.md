# 2026-09-19 — "diverse-6" abundant-style experiment (family-balanced 2/2/2)

**Question.** Companion to the max-dispersion diverse-6 arm (`docs/20260919_diverse6_maxdisp.md`). Max-dispersion
leans NN/Gaba-heavy (those are the most transcriptomically distinct), giving 1 Glut / 2 Gaba / 3 NN. This arm
asks the same 4-hop-vs-2-hop question on a **family-balanced** diverse set — equal representation of the three
cell families — to disentangle "diverse expression profiles" from "balanced taxonomy."

## Selection — family-balanced via max-min dispersion

Of the top-12-by-abundance subclasses, the 6-subset that **maximizes the minimum pairwise distance** between
z-scored expression centroids (a max-min / spread-out-the-closest-pair criterion) lands on a naturally
family-balanced **2 Glut / 2 Gaba / 2 NN** set:

| subclass | family | abundance rank | valtest count |
|---|---|---|---|
| 007 L2/3 IT CTX Glut | Glut | 4 | 6126 |
| 030 L6 CT CTX Glut | Glut | 2 | 7637 |
| 052 Pvalb Gaba | Gaba | 10 | 2119 |
| 053 Sst Gaba | Gaba | 11 | 1538 |
| 319 Astro-TE NN | NN | 3 | 6981 |
| 327 Oligo NN | NN | 7 | 3330 |

Category order (sorted kept, then `other`; codes 0→6):
`007 L2/3 IT` · `030 L6 CT` · `052 Pvalb` · `053 Sst` · `319 Astro-TE NN` · `327 Oligo NN` · `other`.

## Distinctness from the other abundant-style sets

All four sets are distinct (none identical):

| pair | overlap /6 |
|---|---|
| family-balanced vs top-6 | 3 (007, 030, 319) |
| family-balanced vs next-6 | 3 (052, 053, 327) |
| family-balanced vs max-dispersion (`docs/20260919_diverse6_maxdisp.md`) | 5 (differ only: this has 030 L6 CT; maxdisp has 333 Endo) |

So family-balanced is a 3-from-top6 / 3-from-next6 hybrid — different from both submitted abundance sets, and
it differs from the max-dispersion arm by exactly one subclass (030 L6 CT Glut ↔ 333 Endo NN).

## Dataset

Built from the **default 22-label full data** via `gfs/data/build_subclass_variant.py` (same source/split as
the other three sets):

```bash
python -m gfs.data.build_subclass_variant \
  --keep "030 L6 CT CTX Glut,007 L2/3 IT CTX Glut,319 Astro-TE NN,327 Oligo NN,052 Pvalb Gaba,053 Sst Gaba" \
  --out 20260919_diverse6_balanced
```

Output: `data/20260919_diverse6_balanced/isocortex_{train,valtest}_final.h5ad`, 500 genes, **7 labels**.
Build verification: train 74,633 / valtest 63,124; per-class val/test balance ~0.5000, **max |val−test| = 1**;
`other` = 35,393 valtest.

## Protocol

Byte-identical to the abundant/next-6 deep-sample protocol (only path/name differ):
- **2-hop** — `antelope.py`, `n_hops=2`, `trainmode=0`, config `antelope_diverse6_bal.yaml`.
- **4-hop** — `antelope_4hop.py`, `n_hops=4`, `loss_hops=2`, `trainmode=2`, config `antelope_diverse6_bal_4hop.yaml`.
- Both: `n_select=5`, `halfhop=false`, 100 epochs, `lr=1e-3` const, best-`val_f1_overall` ckpt, seeds 0–4.
- Launcher `scripts/main_diverse6_bal_all.sh` — array `0-9%4`; task → `(arm=id/5, seed=id%5)`.

## Status — COMPLETE 2026-09-25

Slurm array `26443774` (`sbatch main_diverse6_bal_all.sh`, submitted 2026-09-19) — **all 10 tasks done**
(2 arms × 5 seeds); last (4-hop seed 4) finished 2026-09-25 11:44. Every run reached epoch 100 with
`test_pred.pt` written. Log dirs `data/logs/*_diverse6_bal_*_nohh_s*/`; stdout `outs/out-comms-26443774_<task>.out`.

## Results — 2026-09-25

F1 recomputed from `test_pred.pt` (argmax of logits vs labels; population sd, ddof=0), paired by seed.
⚠️ n=5 → very low power; read the paired t-test / bootstrap / Cohen dz + sign count together, not any one p.

| metric | 2-hop | 4-hop | Δ (4h−2h) | sign | t 2-sided p | bootstrap 2-sided p | bootstrap 95% CI | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| weighted-F1 | 0.8121 ± 0.0239 | 0.8322 ± 0.0122 | **+0.0202** | 4/5 | 0.121 | **<0.0002** | [+0.0182, +0.0221] | +0.88 |
| macro-F1 | 0.7524 ± 0.0275 | 0.7617 ± 0.0278 | +0.0092 | 3/5 | 0.485 | **<0.0002** | [+0.0065, +0.0119] | +0.34 |

Bootstrap redone 2026-09-25 = **cell-level paired bootstrap**, B=10 000 (`gfs/analysis/paired_cell_bootstrap.py`):
one draw of test-cell indices with replacement, the **same** sample applied to all 5 seeds × both arms,
per-seed F1 recomputed, differenced within seed, then averaged across the 5 seeds; 95% percentile CI +
2-sided p. ⚠️ **Scope: cell-level uncertainty within one biological sample — NOT across-sample/donor
generalization**; far higher-powered than the seed-level paired t-test and can disagree with it. Supersedes the
earlier seed-level bootstrap (B=50 000).

Pairing verified: for each seed, the 2-hop and 4-hop runs score the **identical test set in identical order**
(`test_pred.pt` labels element-wise equal, n=31 562) — so Δ(seed) is a clean matched diff. Caveat on the
replicate unit: the split is **fixed across seeds** (all 5 seeds share the same test-label vector); the seed
randomizes only model init / NeighborLoader sampling / concrete-selection draws, **not the data partition**. So
the n=5 spread (and every seed-level p / CI) reflects training-run stochasticity on one fixed split, not
split/sampling uncertainty.

Per-seed (weighted / macro):

| seed | 2-hop | 4-hop | Δw | Δm |
|---|---|---|---|---|
| 0 | 0.8448 / 0.7789 | 0.8374 / 0.7696 | −0.0074 | −0.0094 |
| 1 | 0.8351 / 0.7678 | 0.8355 / 0.7399 | +0.0004 | −0.0279 |
| 2 | 0.7841 / 0.7001 | 0.8085 / 0.7206 | +0.0244 | +0.0205 |
| 3 | 0.7916 / 0.7625 | 0.8362 / 0.7976 | +0.0446 | +0.0351 |
| 4 | 0.8047 / 0.7529 | 0.8435 / 0.7807 | +0.0388 | +0.0278 |

**Verdict: weighted 4-hop gain (+0.020) is real and consistent in direction, but significance is test-dependent
— treat it as borderline, not a clean win.** The **primary paired t-test calls weighted borderline (p=0.121)
and macro ns (p=0.485)** — 4/5 and 3/5 signs, seed 0 flips negative. The cell-level bootstrap now flags **both**
weighted and macro significant (p<0.0002, CIs entirely above 0), but that is a **within-sample** statement (it
resamples cells from one biological sample and is far higher-powered than the 5-seed t-test); it does **not**
upgrade the cross-seed generalization claim, which the t-test governs → borderline weighted, ns macro. Macro
is dragged down by seed 1 (−0.028) — 053 Sst (smallest kept class) is the least stable. Consistent with the
family-wide pattern: weighted trends up under deep sampling, macro is noisy, n=5 is the binding constraint.
Unlike the max-dispersion arm (weighted 5/5, t p=0.016), balanced does **not** get a clean weighted sweep —
the one substitution (030 L6 CT Glut ↔ 333 Endo NN) matters, and seed 0 flips negative here.

**Panels (final-epoch `selections.csv` → gene_symbol, freq x/5 seeds):**
- 2-hop: Prkcq:5, Ptprm:5, Syt6:4, Zfp536:4, A830036E02Rik:3, Grik3:3, Grm1:2, Lpl:2, Slc7a10:2
- 4-hop: Prkcq:5, Ptprm:5, Slc7a10:4, Syt6:4, Zfp536:4, Grik3:3, A830036E02Rik:2, Grm1:1, Lpl:1, Sox2:1

Core panel (Prkcq, Ptprm, Syt6, Zfp536) is stable across arms; 4-hop leans a bit more on Slc7a10.
