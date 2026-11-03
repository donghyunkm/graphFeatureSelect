# Fine-grained subclass groupings: 2-hop vs 4-hop across Glut / NN / Gaba (enriched label set)

**Date:** 2026-09-14 · **Updated:** 2026-09-20 · **Status:** Gaba fine-grained arm **COMPLETE** (Slurm array
`26380063`, all 30 runs done: fine/cge/mge × 2-hop/4-hop × seeds 0–4, all with `test_pred.pt` at ep100).
**Analysis DONE 2026-09-20** — see the **Results** section below. Verdict: on all three fine Gaba sets, 4-hop
trends up on weighted F1 (all at ceiling ~0.98–0.99) but is **not significant at n=5**; the one substantive
signal is **MGE macro F1 (+0.054, ns), driven by the rare laminar interneuron types** — directionally matching
the CGE-vs-MGE hypothesis but short of significance. Other family (Glut/NN-fine, full-34) and type-2 partition
groupings still PLANNING (not built).

Direct follow-up to `docs/20260907_subclass_grouping_deep_sample.md`. Same experiment (2-hop baseline vs
4-hop complete-field, 5 seeds, select-5, HalfHop-OFF), but on a **new, less-collapsed label set**: many
subclasses that were previously folded into `other` now carry their own label, so each cell-type family is
**finer and harder** (more classes, several very rare). Question: does the cross-family verdict from 09-07
(direction unanimous, only Glut-weighted significant — memory `subclass-grouping-4hop-cross-family`) survive
when each family's label set is enriched with rarer subclasses?

## New dataset (`data/0914_all_subclasses/`, added 2026-09-14 by user)
Same cells / graph / genes / split as the full data, but `obs['subclass']` retains **34 labels** (33 real
subclasses + a now-tiny `other`) instead of 22. "Same data as before, fewer subclasses mapped to `other`."

**Verified (2026-09-14):**
- **34 labels**, 500 genes. train = 74,634 cells, valtest = 63,126 (val 31,563 + test 31,563, **no overlap,
  none unassigned**). Graph present (`obsp['spatial_connectivities']`; valtest 1,262,520 edges).
- **Stratified 50/50 val:test** — every class is within **1 cell** of exact 50/50 (incl. the rarest, e.g.
  BAM 13/13, L6b EPd 13/12).
- `other` is now only **158 train / 100 valtest** cells (was 2,223 / 2,269 in the 09-07 build).
- ⚠️ **Minor discrepancy to confirm:** cell totals are +1 train / +2 valtest vs the 09-07 build
  (74,633 / 63,124 → 74,634 / 63,126). Almost certainly a couple of previously-unlabeled cells now mapped;
  harmless for the experiment (split still clean) but worth the user confirming the provenance.

### Family rosters — new (fine) vs 09-07 (coarse)
Family = the subclass-name suffix (`Glut` / `NN` / `Gaba`). Counts are **train / valtest**.

| family | # subcl (new) | # subcl (09-07) | subclasses newly split out of `other` |
|---|---|---|---|
| **Glut** | **15** (~45.4k) | 10 (kept 9; dropped 020) | 002, 003, 009, **020** (now has valtest), 027, 114 |
| **NN**   | **10** (~22.9k) | 7 | 329 ABC, 332 SMC, 335 BAM |
| **Gaba** | **8** (~6.2k)  | 4 | 047 Sncg, 050 Lamp5 Lhx6, 051 Pvalb chandelier, 056 Sst Chodl |

Full roster with counts (train / valtest):

- **Glut (15):** 001 Car3 981/748 · 002 IT EP-CLA 347/278 · 003 L5/6 IT TPE-ENT 278/163 · 004 L6 IT 4523/4117
  · 005 L5 IT 3080/2993 · 006 L4/5 IT 14185/11387 · 007 L2/3 IT 7885/6126 · 009 L2/3 IT PIR-ENTl 407/313 ·
  020 L2/3 IT RSP 791/478 · 022 L5 ET 2454/2180 · 027 L6b EPd **34/25** · 029 L6b 1197/951 · 030 L6 CT
  8154/7637 · 032 L5 NP 1067/1048 · 114 COAa Barhl2 **35/29**
- **NN (10):** 319 Astro 8955/6981 · 326 OPC 991/777 · 327 Oligo 3718/3330 · 329 ABC 152/215 · 330 VLMC
  647/644 · 331 Peri 954/827 · 332 SMC 389/337 · 333 Endo 5842/5043 · 334 Microglia 1198/993 · 335 BAM **47/26**
- **Gaba (8):** 046 Vip 762/680 · 047 Sncg 211/181 · 049 Lamp5 932/736 · 050 Lamp5 Lhx6 **41/32** · 051 Pvalb
  chandelier **43/32** · 052 Pvalb 2403/2119 · 053 Sst 1691/1538 · 056 Sst Chodl 82/62

⚠️ **Rare-class caveat (important for macro F1):** several new subclasses have **≤65 valtest cells**
(L6b EPd 25, BAM 26, Barhl2 29, Lamp5 Lhx6 32, Pvalb chandelier 32, Sst Chodl 62). A 5-gene panel will almost
certainly miss these → near-zero per-class recall, which **drags and destabilises macro F1**. So on the fine
label sets, weighted F1 stays the robust headline metric; macro will be noisier than in the 09-07 (coarse) run
and its seed-to-seed variance will be larger. This is itself the interesting test: does 4-hop's extra context
help the rare classes at all?

## Groupings we can use
1. **(recommended — "the same experiment") Three fine family groupings.** Keep one family at full resolution,
   collapse the rest into `other`. Direct finer analog of 09-07:

   | dataset | keep | n_labels (kept + other) |
   |---|---|---|
   | **Glut-fine** | 15 Glut subclasses | **16** |
   | **NN-fine**   | 10 NN subclasses   | **11** |
   | **Gaba-fine** | 8 Gaba subclasses  | **9**  |

   Head-to-head with the 09-07 coarse variants (10/7/4-class), this isolates the effect of *label-set
   granularity* on the 4-hop-vs-2-hop gap.
2. **Full fine classification (34-way).** No family collapse — run 2-hop vs 4-hop on all 34 labels at once.
   `other` is negligible (158 cells), so this is effectively full 33-subclass fine typing (cf. the 22-label
   full-data run in memory `4hop-does-not-beat-2hop-full-data`). The hardest single task; good umbrella.
3. **Within-Glut functional cuts** (Glut is now rich enough to sub-group). Either by **projection class** —
   IT {002,003,004,005,006,007,009,020} vs deep/non-IT {022 L5 ET, 027/029 L6b, 030 L6 CT, 032 L5 NP} vs Car3
   {001} — or by **cortical layer** (L2/3 vs L4/5 vs L5 vs L6). Most spatially-laminar axis → strongest test of
   the deeper receptive field. Requires an explicit `--keep`/relabel, not a plain family filter.
4. **Cross-family coarse cuts** (neurons = Glut+Gaba vs NN vs other; or excitatory vs inhibitory). Likely
   near-ceiling like Gaba was (0.984) — low expected information; deprioritise.

### Two *types* of grouping (important)
Everything above (and the 09-07 experiment) is **type 1: "keep a set, collapse the rest into `other`"** — a
one-vs-rest cut, which `build_subclass_variant.py` already supports. But the taxonomy names encode a full
biological hierarchy, so we can also do **type 2: a *partition* — merge every subclass into a coarser
meta-label**, no `other` dumping ground; every cell gets a meaningful class. Type 2 is arguably the more
principled test of "does the graph help at a given biological resolution", and needs only a small builder
extension (a `subclass → meta` **merge-map** instead of a keep-list). Viable type-2 partitions (counts
verified 2026-09-14, train / valtest):

**Glut projection classes (7; the canonical excitatory taxonomy)** — IT 31496/25855 · L6 CT 8154/7637 ·
L5 ET 2454/2180 · L6b {027,029} 1231/976 · L5 NP 1067/1048 · Car3 {001} 981/748 · Barhl2 {114} 35/29
(**fold 114 into `other`/nearest — too rare**). Ties directly to the spatial/laminar axis → strongest
graph-depth test. Can also be coarsened to **IT vs non-IT** (binary) or split by **cortical layer**
(L2/3, L4/5, L5, L6, L6b).

**Gaba by developmental origin (2; very principled)** — CGE-derived {Vip,Sncg,Lamp5} 1905/1597 vs
MGE-derived {Pvalb,Pvalb-chandelier,Sst,Sst-Chodl,Lamp5-Lhx6} 4260/3783. Clean, well-balanced, no rare-class
problem — a nice controlled 2-class contrast.

**Gaba canonical marker classes (5)** — Pvalb {051,052} 2446/2151 · Sst {053,056} 1773/1600 · Lamp5 {049,050}
973/768 · Vip {046} 762/680 · Sncg {047} 211/181. Merges the fine chandelier/Chodl/Lhx6 types up into their
parent marker — mitigates the rare-class issue while staying finer than CGE/MGE.

**NN glia vs vascular/immune (2)** — glia {Astro,OPC,Oligo,Microglia} 14862/12081 vs vascular+immune
{Endo,Peri,SMC,VLMC,BAM,ABC} 8031/7092. Or split further (oligo-lineage {OPC,Oligo}, astro, immune…).

**Global E / I / NN (3; the top-level partition)** — Excitatory(Glut) 45418/38473 · Inhibitory(Gaba)
6165/5380 · Non-neuronal(NN) 22893/19173. The coarsest meaningful task; likely near-ceiling but a clean
sanity anchor.

All type-2 partitions above are well-populated (only Glut-Barhl2 needs folding). They let us ask the 4-hop
question at **several biological resolutions of the same cells** — a cleaner ladder than the abundance- or
family-based type-1 cuts.

## Design (identical to 09-07 — do not change the protocol)
Per dataset: two arms × five seeds (0–4).

| Arm | trainer | n_hops | trainmode | HalfHop | n_select | epochs | lr |
|---|---|---|---|---|---|---|---|
| **2-hop baseline** | `antelope.py` | 2 | 0 | off | 5 | 100 | 1e-3 const |
| **4-hop complete-field** | `antelope_4hop.py` | 4 (loss_hops=2) | 2 | off | 5 | 100 | 1e-3 const |

All other knobs baked into the configs (batch 64, local_layers=2, xyz_status=true, fs_method=persist,
best-`val_f1_overall` ckpt for test). Only `rand_seed` (0–4) varies. Metric of record: `test_f1_overall`
(weighted) + macro, **recomputed from `test_pred.pt`** (memory `recompute-metrics-from-test-pred`), never off a
plot. Paired-by-seed significance per dataset (paired t + Wilcoxon + Cohen dz) — same as 09-07; note **n=5
pins Wilcoxon's minimum p at 0.0625**, so a clean 5/5 win can only clear α=0.05 via the paired t.

## What needs building (nothing done yet)
1. **Family-variant datasets from the NEW source.** `gfs/data/build_subclass_variant.py` currently hard-codes
   reading `<data_root>isocortex_{train,valtest}_final.h5ad` (the 22-label full data). To build the fine family
   variants it must read from `data/0914_all_subclasses/` instead — **add a `--src <subdir>` argument** (default
   = current root behaviour), then:
   ```bash
   python -m gfs.data.build_subclass_variant --src 0914_all_subclasses --family Glut --out 20260914_glut_fine
   python -m gfs.data.build_subclass_variant --src 0914_all_subclasses --family NN   --out 20260914_nn_fine
   python -m gfs.data.build_subclass_variant --src 0914_all_subclasses --family Gaba --out 20260914_gaba_fine
   ```
   (Option 2, full 34-way, needs no build — point configs straight at `0914_all_subclasses`.)
2. **6 configs** `antelope_{glut,nn,gaba}_fine[_4hop].yaml` — clone the 09-07 family configs, change only
   `data.path_{train,valtest}` → the new variant dirs and `data.n_labels` → **16 / 11 / 9**. (For the full
   34-way: `antelope_full34[_4hop].yaml`, n_labels=34, path → `0914_all_subclasses`.)
3. **One array launcher** mirroring `scripts/main_subclass_grouping_all.sh` (`--array=0-29%4`, task id →
   dataset/arm/seed). Reuse the same task-id math.
4. Smoke-test one 2-hop + one 4-hop build+few-steps before launching the full array (per CLAUDE.md).

## Recommendation
Run **option 1 (three fine family groupings)** — it is literally "the same experiment" on the enriched labels
and gives the cleanest apples-to-apples read on whether granularity changes the 4-hop verdict. Optionally add
**option 2 (full 34-way)** as a single extra dataset if we want the hardest-case umbrella (would extend the
array to 0–39). Hold options 3–4 unless option 1's result motivates them.

## Gaba fine-grained run — RUNNING (submitted 2026-09-14, Slurm array `26380063`)
First concrete experiment off this dataset: three Gaba keep-sets (fine subclasses kept at full resolution,
everything else → `other`), 2-hop vs 4-hop, seeds 0–4 — same protocol as 09-07. Motivation: the 09-07 Gaba
run was **ceiling-bound** (only 4 easy canonical classes → 0.984 wtd, no headroom, 4-hop not significant).
The enriched set adds the *hard* interneuron subtypes, creating real macro-F1 headroom where a spatial graph
should matter.

### Developmental origin (why CGE vs MGE)
Cortical GABAergic interneurons are born in the ventral forebrain ganglionic eminences and migrate into cortex;
origin predicts marker, morphology, function, and **laminar position** (why it's relevant to a spatial-graph model):
- **MGE** (medial GE; *Nkx2-1→Lhx6*): **Pvalb** (fast-spiking basket + chandelier), **Sst** (Martinotti; incl.
  Sst-Chodl), and the **Lamp5-Lhx6** type. Enriched in mid–deep layers (L4–6).
- **CGE** (caudal GE; *Prox1*/5-HT3aR): **Vip** (disinhibitory), **Lamp5** (neurogliaform), **Sncg** (CCK basket).
  Enriched superficially (L1–3).

So CGE vs MGE is both a marker-defined and a laminar contrast; comparing them tells us whether 4-hop helps more
within the laminarly-structured MGE lineage than the superficial CGE one.

### Datasets built (from `data/0914_all_subclasses/` via `build_subclass_variant.py --src`; balance verified, max |val−test| = 1)
| dataset dir | kept fine subclasses | n_labels | kept vt / other vt |
|---|---|---|---|
| `20260914_gaba_fine` | all 8 Gaba | 9 | 5,380 / 57,746 |
| `20260914_gaba_cge`  | 046 Vip, 047 Sncg, 049 Lamp5 | 4 | 1,597 / 61,529 |
| `20260914_gaba_mge`  | 050 Lamp5-Lhx6, 051 Pvalb-chandelier, 052 Pvalb, 053 Sst, 056 Sst-Chodl | 6 | 3,783 / 59,343 |

Builder gained a **`--src <subdir>`** arg (default = 22-label root) so it can read the enriched file:
```bash
python -m gfs.data.build_subclass_variant --src 0914_all_subclasses --family Gaba --out 20260914_gaba_fine
python -m gfs.data.build_subclass_variant --src 0914_all_subclasses --keep "046 Vip Gaba,047 Sncg Gaba,049 Lamp5 Gaba" --out 20260914_gaba_cge
python -m gfs.data.build_subclass_variant --src 0914_all_subclasses --keep "050 Lamp5 Lhx6 Gaba,051 Pvalb chandelier Gaba,052 Pvalb Gaba,053 Sst Gaba,056 Sst Chodl Gaba" --out 20260914_gaba_mge
```

### Configs + launcher
- 6 configs `antelope_gaba_{fine,cge,mge}[_4hop].yaml` (cloned from the 09-07 gaba configs; only `path_*`,
  `n_labels`, `name` changed — protocol otherwise identical: select-5, HalfHop-off, 100 ep, best-val ckpt).
- **One throttled array** `scripts/main_gaba_grouping_all.sh` (`--array=0-29%4`, ≤4 concurrent). Task id →
  (dataset, arm, seed): dataset = id/10 (0=fine, 1=cge, 2=mge), arm = (id%10)/5 (0=2-hop, 1=4-hop), seed = id%5.
- **Smoke-tested (CPU, 2026-09-14):** all 3 datasets load; `celltype` labels in `[0, n_labels−1]` (9/4/6);
  `model.forward` logits match out_ch; 4-hop `loss_mask` present + non-empty (~4.6k nodes). ruff clean.

**Launched:** `sbatch main_gaba_grouping_all.sh` → **array `26380063`** (2026-09-14, from `scripts/`).
Monitor: `squeue -u donghyun.kim` / `outs/out-comms-26380063_<task>.out`. Log dirs
`data/logs/<datetime>_gaba_{fine,cge,mge}_{seed,4hop_seed}<s>_nohh_s<s>/`.

## Results _(recomputed from `test_pred.pt`, n=31,563 test cells; population sd, ddof=0; paired by seed; 2026-09-20)_

All 30 runs complete (verified: every fine/cge/mge × 2-hop/4-hop × seed 0–4 has `test_pred.pt` at epoch 100).
Metrics via `gfs/analysis/gaba_grouping_f1.py` (argmax of logits vs labels; `sklearn` weighted/macro F1).
Paired by seed — the val/test split is identical across seeds (masks copied verbatim), so 2-hop seed_i and
4-hop seed_i score the exact same test cells. n=5 pins the Wilcoxon minimum p at 0.0625, so only the paired t
can clear α=0.05.

| dataset | metric | 2-hop (mean±sd) | 4-hop (mean±sd) | Δ (4−2) | per-seed Δ | pos | paired t p | Wilcoxon p | bootstrap p (95% CI) | Cohen dz |
|---|---|---|---|---|---|---|---|---|---|---|
| **fine** (9 lbl) | weighted | 0.9828 ± 0.0012 | 0.9837 ± 0.0005 | **+0.0009** | [−.0013,+.0029,+.0026,+.0010,−.0007] | 3/5 | 0.35 (ns) | 0.44 | 0.0132 [+.0002,+.0016] | +0.47 |
| **fine** (9 lbl) | macro | 0.6745 ± 0.0269 | 0.6748 ± 0.0149 | +0.0003 | [−.0147,−.0017,+.0423,−.0075,−.0169] | 1/5 | 0.98 (ns) | 0.625 | 0.96 [−.0194,+.0211] ns | +0.01 |
| **cge** (4 lbl) | weighted | 0.9918 ± 0.0006 | 0.9928 ± 0.0010 | **+0.0009** | [+.0014,+.0004,−.0007,+.0023,+.0012] | 4/5 | 0.14 (ns) | 0.19 | 0.0002 [+.0005,+.0014] | +0.82 |
| **cge** (4 lbl) | macro | 0.8108 ± 0.0148 | 0.8100 ± 0.0546 | −0.0007 | [+.0484,−.0305,−.0691,+.0222,+.0254] | 3/5 | 0.97 (ns) | 1.00 | 0.90 [−.0145,+.0138] ns | −0.02 |
| **mge** (6 lbl) | weighted | 0.9901 ± 0.0005 | 0.9907 ± 0.0006 | **+0.0005** | [+.0007,−.0000,−.0003,+.0011,+.0013] | 3/5 | 0.15 (ns) | 0.31 | 0.0036 [+.0002,+.0009] | +0.80 |
| **mge** (6 lbl) | macro | 0.6454 ± 0.0537 | 0.6995 ± 0.0560 | **+0.0541** | [+.1217,−.0076,−.0070,+.1217,+.0418] | 3/5 | 0.14 (ns) | 0.31 | **<0.0002** [+.0356,+.0722] | +0.83 |

Bootstrap redone 2026-09-25 = **cell-level paired bootstrap**, B=10 000 (`gfs/analysis/paired_cell_bootstrap.py`):
shared test-cell resample across all 5 seeds × both arms, per-seed F1 differenced within seed then averaged; 95%
percentile CI + 2-sided p. ⚠️ **Scope: cell-level uncertainty within one biological sample — NOT across-sample/donor
generalization**; far higher-powered than the seed-level paired t-test and can disagree with it. Under it, all three
**weighted** Δ's are significant (though ceiling-tiny, +0.0005 to +0.0009) and **MGE macro is strongly significant**
(+0.0541, p<0.0002), while **fine-macro and cge-macro remain ns** (CIs cross zero). The **paired t-test stays the
primary/reported test** — nothing clears α=0.05 on it at n=5. Supersedes the earlier seed-level bootstrap (B=50 000).

**Nothing reaches α=0.05 on the primary (t) test.** Every weighted F1 is at ceiling (0.983–0.993) with essentially no headroom, so the
4-hop weighted Δ is tiny (+0.0005 to +0.0009) though directionally consistent (dz 0.47–0.82, positive in 3–4/5).
Macro is the informative axis, and it splits by dataset:
- **fine (all 8 Gaba subclasses):** dead tie on macro (Δ+0.0003, dz≈0). Only seed2 shows a 4-hop macro gain
  (+0.042); the other four are flat-to-negative. Enriching the full Gaba set did not surface a 4-hop win.
- **cge (Vip/Sncg/Lamp5, superficial):** macro tie (Δ−0.0007) with **hugely inflated 4-hop variance**
  (±0.0546 vs 2-hop's ±0.0148) — seed0 jumps to 0.885 while seed2 collapses to 0.725. Unstable, no signal.
- **mge (Pvalb/Sst/chandelier/Chodl/Lhx6, mid-deep laminar):** the **largest macro gain, +0.0541** (dz=0.83),
  ns on the primary t-test at n=5 — driven almost entirely by seeds 0 and 3 (+0.122 each; seed3's 2-hop macro
  was a 0.544 dip). The cell-level paired bootstrap **strongly backs this signal (p<0.0002, CI [+0.036,+0.072])**
  as a within-sample statement — but that measures cell-level uncertainty on this one sample, not cross-seed/donor
  generalization, so the t-ns call still governs the generalization claim.

### MGE per-class recall — where the macro gain lives (mean over 5 seeds)
Label codes (by test count): 0=Lamp5-Lhx6 (n=16), 1=Pvalb-chandelier (16), 2=Pvalb (1060), 3=Sst (769),
4=Sst-Chodl (31), 5=other (29671).

| class | 2-hop recall | 4-hop recall | Δ |
|---|---|---|---|
| Lamp5-Lhx6 (050, rare) | 0.013 | 0.062 | +0.049 |
| Pvalb-chandelier (051, rare) | 0.762 | 0.875 | **+0.113** |
| Pvalb (052) | 0.917 | 0.909 | −0.008 |
| Sst (053) | 0.917 | 0.905 | −0.012 |
| **Sst-Chodl (056, rare)** | 0.219 | 0.406 | **+0.187** |
| other | 0.997 | 0.997 | 0.000 |

The MGE macro lift comes **entirely from the rare laminar interneuron types** (Sst-Chodl +0.19, chandelier
+0.11, Lamp5-Lhx6 +0.05); the abundant Pvalb/Sst dip marginally. This is exactly the mechanism the experiment
was designed to test — 4-hop's deeper receptive field helps the hard, sparse, spatially-structured MGE subtypes
— but the effect is noisy (Lamp5-Lhx6 at 6% recall is near-useless; a 5-gene panel misses it either way) and
does not clear significance at n=5.

### CGE vs MGE (the laminar/developmental-origin contrast)
Hypothesis (see "Developmental origin" above): 4-hop should help the laminarly-structured **MGE** lineage more
than the superficial **CGE** lineage. **Result: directionally confirmed but not significant.** MGE macro
Δ=**+0.0541** vs CGE macro Δ=**−0.0007** — the entire macro benefit is on the MGE side, consistent with a
spatial/laminar mechanism. But CGE's macro is destabilised by variance rather than genuinely flat, both
weighted axes are at ceiling, and neither dataset's Δ passes α=0.05 at n=5, so this is a suggestive trend, not
an established result. More seeds would be needed to claim it.

### Comparison to 09-07 coarse Gaba
09-07 coarse Gaba (4 canonical classes, ceiling-bound): weighted 0.9834→0.9844, macro 0.8841→0.8931, macro
won **5/5 seeds** (p=0.066). Enriching the label set **lowered macro sharply** (fine 0.67, mge 0.65, cge 0.81
vs coarse 0.88) — confirming the rare-class caveat: the added hard subtypes (chandelier/Chodl/Lhx6, ≤32 vt)
drag and destabilise macro exactly as predicted (doc lines 48–53). The 4-hop macro signal is **weaker and
noisier** than the coarse run (coarse 5/5 clean; here fine 1/5, cge 3/5, mge 3/5 — none clean). So finer
granularity did **not** strengthen the 4-hop verdict; it mostly added rare-class noise.

### Gene panels (final-epoch hard selection, `selections.csv` last row; symbol → freq across 5 seeds)
| dataset | arm | panel selection frequency | backbone / shift |
|---|---|---|---|
| **fine** | 2-hop | `Elfn1`×5, `Nxph1`×5, `Gad2`×3, `Oprd1`×3, `Slc32a1`×3, `Synpr`×2 | backbone `{Elfn1, Nxph1}` |
| **fine** | 4-hop | `Oprd1`×5, `Synpr`×5, `Gad2`×3, `Slc32a1`×3, `Elfn1`×2, `Kit`×2, `Nxph1`×2 | swaps to `{Oprd1, Synpr}`; drops `Nxph1` (5→2), `Elfn1` (5→2) |
| **cge** | 2-hop | `Igf1`×5, `Kit`×5, `Grik3`×3, `Npas1`×3, `Vip`×3 | backbone `{Igf1, Kit}` |
| **cge** | 4-hop | `Kit`×4, `Oprd1`×4, `Vip`×4, `Gad2`×2, `Grpr`×2, `Npas1`×2, `Slc32a1`×2 | keeps `Kit`; drops `Igf1` (5→1), promotes `Oprd1`+`Vip` |
| **mge** | 2-hop | `Elfn1`×5, `Nxph1`×5, `Oprd1`×5, `Synpr`×4, `Kcnk9`×2 | backbone `{Elfn1, Nxph1, Oprd1}` |
| **mge** | 4-hop | `Elfn1`×5, `Nxph1`×5, `Oprd1`×5, `Synpr`×5, `Kcnk9`×3 | **same panel**; only firms `Synpr` (4→5), `Kcnk9` (2→3) |

- **fine** shows the clearest depth-driven swap: 2-hop locks `{Elfn1, Nxph1}`, 4-hop locks `{Oprd1, Synpr}`
  instead — the **same `Nxph1`→`Synpr` trade** seen in the 09-07 coarse Gaba run (which added `Synpr` 1/5→5/5),
  here amplified into dropping both `Nxph1` and `Elfn1` for `{Oprd1, Synpr}`. `Gad2` (canonical GABA marker) and
  `Slc32a1` (VGAT) sit at 3/5 in both arms.
- **mge** is the opposite extreme — **both arms converge on the identical panel** `{Elfn1, Nxph1, Oprd1, Synpr}`
  + `Kcnk9`; depth barely reshapes it (only firms up `Synpr`/`Kcnk9`). The MGE lineage has a stable, well-defined
  marker set that 2-hop already finds, so 4-hop's gain (the rare-class recall lift above) comes from *using* the
  graph better, not from picking different genes.
- **cge** keeps `Kit` but is otherwise the least stable (matching its inflated F1 variance): 4-hop drops the
  strong 2-hop `Igf1` (5/5→1/5) and scatters into `{Oprd1, Vip, Grpr, Npas1, Slc32a1}`. Same "depth degrades
  panel stability" pattern NN showed at 09-07.

### Bottom line
Enriching the Gaba label set with the hard rare interneuron subtypes did **not** produce the clean 4-hop win the
experiment was designed to find. 4-hop is directionally ≥ 2-hop everywhere (all weighted Δ positive, MGE macro
clearly up) — consistent with the 09-07 cross-family verdict and `subclass-grouping-4hop-cross-family` — but
**nothing clears α=0.05 at n=5**, weighted is uniformly at ceiling, and the one real macro signal (MGE, rare
laminar types) is noisy. The CGE-vs-MGE laminar contrast points the predicted way (MGE macro +0.054 vs CGE
−0.001) but is a trend, not a result. More seeds would be required to establish either claim.

## Status / next steps
- [x] Add `--src` to `build_subclass_variant.py`; build the **3 Gaba** variants (fine/cge/mge) — done, balance
      verified (max |val−test| = 1). *(Glut-fine/NN-fine/full-34 variants NOT built — deferred.)*
- [x] Add the **6 Gaba configs** `antelope_gaba_{fine,cge,mge}[_4hop].yaml`; add the array launcher
      `scripts/main_gaba_grouping_all.sh`; CPU smoke-test (all 3 datasets load, labels 9/4/6, forward OK,
      4-hop loss_mask non-empty). *(Glut/NN-fine + full-34 configs NOT added.)*
- [x] User `sbatch`ed the array → **`26380063` RUNNING** (2026-09-14 07:27). Confirmed exact 09-07 protocol
      (4-hop tm2 vs 2-hop tm0, select-5, HalfHop-off, 100 ep, best-val ckpt); config diff byte-identical to 09-07
      except path/n_labels/name.
- [x] **DONE 2026-09-20 — `26380063` complete (30/30):** recomputed weighted + macro F1 (from `test_pred.pt`,
      population sd) + paired-by-seed significance (t / Wilcoxon / Cohen dz) + gene panels for gaba_fine/cge/mge
      via `gfs/analysis/gaba_grouping_f1.py`; compared to 09-07 coarse Gaba; did the CGE-vs-MGE laminar contrast
      and MGE per-class recall. See the **Results** section above. Verdict: 4-hop directionally ≥ 2-hop
      everywhere but nothing significant at n=5; weighted at ceiling; MGE macro +0.054 (rare laminar types) is
      the one real signal (ns); CGE-vs-MGE points the predicted way but is a trend. Rare-class caveat confirmed
      (macro dropped 0.88→0.65–0.81 vs coarse).
- [ ] (Optional) Confirm the +1/+2 cell-count provenance vs the 09-07 build with the user (harmless).
- [ ] (Deferred) Glut-fine (16) / NN-fine (11) / full-34 groupings + type-2 partitions (Glut projection classes,
      NN glia-vs-vascular, E/I/NN) — build only if the Gaba result motivates them.

### Verify results when the array finishes
```bash
source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh && conda activate gfs
# per-task test_pred.pt lands in data/logs/<datetime>_gaba_{fine,cge,mge}_{seed,4hop_seed}<s>_nohh_s<s>/
# recompute from o["predictions"] (logits [N,C]) + o["labels"]; gene idx→symbol via adata.var["gene_symbol"]
```
