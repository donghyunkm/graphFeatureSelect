# Complete-receptive-field training (4-hop over-sampling)

**Date:** 2026-08-20 (2x2 completed 2026-08-22; HalfHop-OFF arms extended to seeds 0–4 on 2026-08-25) ·
**Status:** full 2x2 ablation complete (seed 0); HalfHop-OFF comparison now has 5-seed error bars.

**The idea in one line:** over-sample each training patch to 4 hops, then take the loss *only* on the
subset of nodes whose full 2-hop receptive field was actually sampled — so every node the model is trained
on has a correct embedding. (Previously written as "deep-sample / shallow-loss"; same method.)

## Summary

We find that the 4hop model (Complete-receptive-field training) performs better than the 2hop model when trained/evaluated on the raw graph (no half-hop augmentation). Importantly, it also selects a different gene panel. The table below is seed 0; the gain replicates across seeds 0–4 (weighted 0.7851 ± 0.0073 vs 0.7719 ± 0.0043, 4hop > 2hop on every seed — see "Multi-seed error bars" below).

| Model | F1 (w / m) | Gene 1 | Gene 2 | Gene 3 | Gene 4 | Gene 5 |
|---|---|---|---|---|---|---|
| 2hop/tm0, HalfHop OFF | 0.7750 / 0.7603 | Rorb (38) | A830036E02Rik (83) | Dscaml1 (480) | Grik3 (2) | Hmcn1 (325) |
| 4hop/tm2, HalfHop OFF | 0.7933 / 0.7824 | Rorb (38) | A830036E02Rik (83) | Dscaml1 (480) | Syt6 (5) | Ptprm (12) |


When HalfHop ON, we find that the 4hop model does worse than the 2hop model. 

This is a reasonable result as half-hop augmentation changes the graph topology before message passing. Half-hop modifies a graph by inserting an intermediate node along selected edges, turning an original edge u↔v into u↔w↔v. This changes message passing because directly connected original nodes no longer exchange information in a single GNN layer; their information must pass through the inserted node. 


## Motivation
The abundant-subclass experiment runs `trainmode=0`: training loss is computed on **every** node in
the sampled patch (`train_mask = ones`, ~4147 nodes/batch). But `NeighborLoader` with
`num_neighbors=[-1]*2` only guarantees a complete 2-hop receptive field for the **seed/root** node.
With a 2-layer GNN (`local_layers=2`):

- root (hop 0): needs 2 hops → complete ✓
- hop-1 node: its 2-layer output needs up to hop-3 from the seed → **not sampled** ✗
- hop-2 node: needs up to hop-4 → **very incomplete** ✗

So under `trainmode=0` only the root has a correct embedding; all other loss nodes are trained on
truncated neighborhoods. (Eval was always fine — val/test use root-only.)

## Fix
Sample **`n_hops=4`** during training but compute loss only on nodes within **`loss_hops=2`** of a
seed. With 4-hop sampling, every hop-≤2 node has its full 2-hop closure materialised, so its embedding
is exact. Hops 3–4 exist only to complete those fields; they never enter any loss node's 2-layer
receptive field (message-passing depth = `local_layers`, independent of sampling depth), and no loss
is taken on them.

Invariants: `loss_hops == model.local_layers` (=2) and `n_hops == 2*loss_hops` (=4). Val/test sample
`loss_hops` hops (enough for a complete root field) and **evaluate root nodes only** — unchanged.

## Implementation
- `gfs/data/hemisphere_new_4hop.py` — `NeighborLoaderMod` now takes `(n_hops, loss_hops)`: assigns
  `subgraph_id` over the full `n_hops` patch (so every loaded node gets a gene mask) and attaches
  `batch.loss_mask` via a cheap collective `loss_hops`-hop BFS from all seeds. Val/test loaders sample
  `loss_hops` hops.
- `gfs/models/antelope.py` — new `trainmode==2` branch: `idx = torch.where(batch.loss_mask)[0]`.
  Modes 0 (all nodes) / 1 (root only) untouched.
- `gfs/configs/antelope_abundant_4hop.yaml` — `n_hops=4`, `loss_hops=2`, `trainmode=2`, `batch_size=64`.
- `gfs/trainers/antelope_4hop.py` — imports the new data module, passes `loss_hops`.
- `scripts/main_abundant_4hop.sh` — launcher (`n_select=5`, `seed=0`, prefix `abundant_4hop_seed0`).

## Smoke-test numbers (measured 2026-08-20)
- avg directed degree ≈ 21 (1,567,314 edges / 74,634 train nodes).
- Spatial locality dampens the 4-hop blow-up hard (nowhere near full-graph): bs=16 → ~3.7k nodes/batch,
  ~1,179 loss nodes; **bs=64 → ~14k nodes / ~177k edges/batch, ~4,650 complete-field loss nodes**, ~1.7 s/batch load.
- Forward+backward verified: `pred[idx]` aligns with labels, all nodes get a `subgraph_id`, gradient
  reaches the selection logits.

## Comparison arm (batch_size=64 — matches baseline)
Direct successor to `20260811_abundant_subclass.md` (same data, `n_select=5`, seed 0, 100 epochs). At
`bs=64` this arm matches the baseline's steps/epoch (~1,166 vs 1,167) and loss-node volume
(~4,650 vs ~4,147) — the only intended differences are 4-hop sampling + `trainmode=2` (complete
receptive fields). Compare its panel + root-only test F1 against that doc's intact `{2,12,38,83,480}` /
0.7789 weighted. NOTE: eval-time HalfHop is left ON in both arms (unchanged), so the comparison stays fair.

## HalfHop 2x2 ablation
`model.halfhop=false` (config flag, no code change) turns HalfHop off for BOTH train and eval — the
latter removes the eval-time graph perturbation (see `## The wrinkle` in the discussion), so root eval
message-passes over the raw 2-hop graph. Scripts `main_abundant_nohh.sh` and `main_abundant_4hop_nohh.sh`
add the HalfHop-OFF row. Full 2x2 (seed 0, n_select=5, bs=64):

**Results** (seed 0, n_select=5, bs=64, n=31,563; F1 recomputed from `test_pred.pt`, weighted / macro):

| | HalfHop ON | HalfHop OFF |
|---|---|---|
| 2-hop / trainmode=0 (baseline) | **0.7789** / 0.7618 | 0.7750 / 0.7603 |
| 4-hop / trainmode=2 (complete-field loss) | 0.7671 / 0.7563 | **0.7933 / 0.7824** |

Jobs: 2hop/HHon = prior (`20260810_204705_..._s0`); 2hop/HHoff = 25273001; 4hop/HHon = 25272602;
4hop/HHoff = 25273003. All ran 100 epochs, tested from best-`val_f1_overall` checkpoint.

### Multi-seed error bars — HalfHop OFF comparison (added 2026-08-25)

Seeds 1–4 completed (arrays 2hop = `25336508`, 4hop = `25336509`), giving 5 seeds (0–4) per HalfHop-OFF
arm. Same data, `n_select=5`, `bs=64`, 100 epochs, tested from best-`val_f1_overall` checkpoint; F1
recomputed from `test_pred.pt` (matches printed Lightning tables to 4 dp), n=31,563 test cells. Every run's
config verified from `hparams.yaml` (2hop: `n_hops=2`/`trainmode=0`; 4hop: `n_hops=4`/`loss_hops=2`/`trainmode=2`; `halfhop=false` throughout).

| Arm (HalfHop OFF) | weighted F1 (mean ± sd) | macro F1 (mean ± sd) |
|---|---|---|
| 2-hop / tm0 | 0.7719 ± 0.0043 | 0.7583 ± 0.0054 |
| 4-hop / tm2 (complete-field) | **0.7851 ± 0.0073** | **0.7764 ± 0.0067** |
| Δ (4hop − 2hop) | **+0.0132** | **+0.0181** |

Per-seed weighted / macro F1:

| seed | 2-hop w / m | 4-hop w / m |
|---|---|---|
| 0 | 0.7750 / 0.7603 | 0.7933 / 0.7824 |
| 1 | 0.7647 / 0.7499 | 0.7792 / 0.7698 |
| 2 | 0.7744 / 0.7627 | 0.7876 / 0.7794 |
| 3 | 0.7712 / 0.7559 | 0.7895 / 0.7818 |
| 4 | 0.7742 / 0.7624 | 0.7759 / 0.7684 |

**The complete-field gain replicates on every seed** (4-hop > 2-hop for all 5), weighted +1.3 pts / macro
+1.8 pts. The gap exceeds the combined per-arm spread, so the seed-0 result is not a fluke. Scoped to
raw-graph (HalfHop-OFF) eval — the HalfHop-ON sign-flip has not yet been given seeds 1–4.

**Cell-level paired bootstrap (added 2026-09-25).** B=10 000, `gfs/analysis/paired_cell_bootstrap.py`:
each replicate draws test-cell indices with replacement, applies the **same** sample to all 5 seeds ×
both arms, recomputes per-seed F1, differences within seed, and averages the 5 paired diffs; 95%
percentile CI + two-sided p.

| metric | Δ (4h−2h) | 95% CI | bootstrap 2-sided p |
|---|---|---|---|
| weighted | +0.0132 | [+0.0108, +0.0157] | <0.0002 |
| macro | +0.0181 | [+0.0154, +0.0208] | <0.0002 |

Both CIs sit well above zero (0/10 000 replicates crossed) — the win is significant on this sample, and
here it agrees with the 5/5 per-seed sweep. ⚠️ **Scope: this is cell-level uncertainty within one
biological sample — NOT across-sample/donor generalization.** The cell-level bootstrap is far higher-powered
than a seed-level paired test and can disagree with it; a seed-level paired t-test remains the primary test
for the cross-seed generalization claim.

#### Per-class accuracy — where the 4-hop gain comes from (added 2026-09-07)

Per-class accuracy = recall (of the cells truly of class *c*, the fraction predicted *c*), recomputed from
`test_pred.pt` (argmax vs `labels`), mean ± sd over seeds 0–4, same 10 HalfHop-OFF runs as above
(n=31,563 test cells; support sums to 31,563). Class code order = the abundant `subclass` categories
(sorted kept subclasses, then `other`).

| class (subclass) | test support | 2-hop acc | 4-hop acc | Δ (4−2) |
|---|---|---|---|---|
| 004 L6 IT CTX Glut | 2058 | 0.4998 ± 0.0302 | 0.6006 ± 0.0299 | **+0.1008** |
| 006 L4/5 IT CTX Glut | 5693 | 0.8503 ± 0.0097 | 0.8458 ± 0.0098 | −0.0044 |
| 007 L2/3 IT CTX Glut | 3063 | 0.8463 ± 0.0224 | 0.8219 ± 0.0324 | −0.0244 |
| 030 L6 CT CTX Glut | 3819 | 0.8453 ± 0.0340 | 0.9018 ± 0.0083 | **+0.0565** |
| 319 Astro-TE NN | 3491 | 0.7020 ± 0.0245 | 0.6844 ± 0.0625 | −0.0176 |
| 333 Endo NN | 2522 | 0.7729 ± 0.0256 | 0.7668 ± 0.0173 | −0.0061 |
| other | 10917 | 0.7617 ± 0.0117 | 0.7746 ± 0.0227 | +0.0129 |
| **balanced acc** (mean per-class recall) | 31563 | 0.7540 | **0.7709** | **+0.0168** |

Reproduce this table: `python -m gfs.analysis.per_class_accuracy` (log dirs + `DATA_SUBDIR` hard-coded to
these 10 runs; edit `ARMS` for other experiments).

**The 4-hop win is concentrated in the deep-layer L6 Glut classes.** The two largest gains are
`004 L6 IT CTX Glut` (+0.101, the weakest class in both arms) and `030 L6 CT CTX Glut` (+0.057) — the
deepest cortical layers, exactly where a larger receptive field should help most. `other` also rises
(+0.013). The remaining four classes drift slightly *down* (−0.004 to −0.024) but all within ~1 sd, so
4-hop trades a small, spread-out cost against two large deep-layer gains. This dovetails with the panel
result — 4-hop reliably swaps in `Syt6`(5), a deep-layer (L6) marker, in place of `Grik3`(2).

#### Panels across seeds (HalfHop OFF)

Selection read at each run's **best-`val_f1_overall` checkpoint epoch** (2hop epochs 89/90/81/90/93;
4hop 88/82/83/96/96), `n_select=5`, symbols from `data/0810_abudant_classes_only_top6/*.h5ad` `var` symbol column.

| seed | 2-hop / tm0 panel | 4-hop / tm2 panel |
|---|---|---|
| 0 | Grik3, Rorb, A830036E02Rik, Hmcn1, Dscaml1 | Syt6, Ptprm, Rorb, A830036E02Rik, Dscaml1 |
| 1 | Grik3, Rorb, A830036E02Rik, Hmcn1, Dscaml1 | Syt6, Rorb, A830036E02Rik, Hmcn1, Dscaml1 |
| 2 | Grik3, Ptprm, Rorb, A830036E02Rik, Hmcn1 | Grik3, Syt6, Rorb, A830036E02Rik, Hmcn1 |
| 3 | Grik3, Rorb, A830036E02Rik, Hmcn1, Dscaml1 | Syt6, Slc7a10, A830036E02Rik, Hmcn1, Dscaml1 |
| 4 | Syt6, Rorb, A830036E02Rik, Hmcn1, Dscaml1 | Grik3, Syt6, A830036E02Rik, Hmcn1, Dscaml1 |

Selection frequency (times chosen / 5 seeds; gene-column index in parens):

| gene | idx | 2-hop | 4-hop |
|---|---|---|---|
| A830036E02Rik | 83 | 5 | 5 |
| Hmcn1 | 325 | 5 | 4 |
| Dscaml1 | 480 | 4 | 4 |
| Rorb | 38 | 5 | 3 |
| Grik3 | 2 | 4 | 2 |
| Syt6 | 5 | 1 | 5 |
| Ptprm | 12 | 1 | 1 |
| Slc7a10 | 45 | 0 | 1 |

**Panel findings (multi-seed).**
1. **Stable backbone across both arms:** `A830036E02Rik`(83) is in all 10 runs; `Hmcn1`(325) 9/10; `Dscaml1`(480) 8/10.
2. **The depth signature is `Syt6`(5) ⇄ `Grik3`(2):** `Syt6` is in **all 5** 4-hop seeds but only 1/5 at 2-hop;
   `Grik3` is the mirror (4/5 at 2-hop, 2/5 at 4-hop). Confirms the seed-0 "variable slot tracks sampling depth"
   observation robustly — deeper sampling systematically swaps `Grik3` → `Syt6`.
3. `Rorb`(38) weakens slightly at 4-hop (5/5 → 3/5); one-off picks are `Ptprm`(12, once per arm) and `Slc7a10`(45, 4-hop seed 3 only).
4. **Amends the seed-0 core claim:** across seeds `Hmcn1`(325) is more consistent than `Rorb`(38) in the 4-hop
   arm, so the durable multi-seed backbone is `{A830036E02Rik, Hmcn1, Dscaml1}` (+ `Rorb` at 2-hop), not the
   seed-0 `{Rorb, A830036E02Rik, Dscaml1}`.

Selected genes (5 per model; selections stable over last 5 epochs; symbol + gene-column index):

| Model | F1 (w / m) | Gene 1 | Gene 2 | Gene 3 | Gene 4 | Gene 5 |
|---|---|---|---|---|---|---|
| 2hop/tm0, HalfHop ON (baseline) | 0.7789 / 0.7618 | Rorb (38) | A830036E02Rik (83) | Dscaml1 (480) | Grik3 (2) | Ptprm (12) |
| 2hop/tm0, HalfHop OFF | 0.7750 / 0.7603 | Rorb (38) | A830036E02Rik (83) | Dscaml1 (480) | Grik3 (2) | Hmcn1 (325) |
| 4hop/tm2, HalfHop ON | 0.7671 / 0.7563 | Rorb (38) | A830036E02Rik (83) | Dscaml1 (480) | Syt6 (5) | Hmcn1 (325) |
| 4hop/tm2, HalfHop OFF (best) | 0.7933 / 0.7824 | Rorb (38) | A830036E02Rik (83) | Dscaml1 (480) | Syt6 (5) | Ptprm (12) |

Genes 1–3 are the shared **core in all four** (`Rorb`(38), `A830036E02Rik`(83), `Dscaml1`(480)); genes 4–5
are the variable slots. Symbols mapped from `data/0810_abudant_classes_only_top6/*.h5ad` `var["gene_symbol"]`.

The swaps are structured: **variable slot A tracks sampling depth** (2-hop→`Grik3`, 4-hop→`Syt6`,
independent of HalfHop). **Variable slot B tracks F1** — `Ptprm` in the two higher-F1 arms, `Hmcn1` in the
two lower (XOR over the 2x2). So sampling depth does perturb *selection* (unlike the graph-invariant
intact-vs-shuffle result), but only the 5th slot and toward a plausible deep-layer marker.

**Findings.**
1. **Best cell = 4-hop complete-field loss + HalfHop OFF: 0.7933 w / 0.7824 m**, beating the prior 0.7789
   baseline by +0.014 weighted / +0.021 macro. Correcting the incomplete-receptive-field bug (tm0 takes loss
   on partial-field nodes → tm2 takes loss only on complete-field nodes) genuinely helps — **but only on the
   raw graph.**
2. **HalfHop's effect flips sign with sampling depth.** At 2-hop it barely helps (+0.004 w). At 4-hop it
   *hurts substantially* (−0.026 w: 0.7671 vs 0.7933). The 4-hop patch is far denser (~14k nodes), so
   HalfHop's slow-node insertions over-perturb message passing; the complete-field win only materializes with
   HalfHop off. Caveat: the two ON arms carry eval-time HalfHop (perturbed-graph eval); the OFF arms are
   raw-graph eval — the flip conflates train-time and eval-time HalfHop, not yet separated.

## Not yet done
- Seeds 1–4 for the HalfHop sign-flip (only the HalfHop-OFF comparison has 5-seed error bars so far —
  done 2026-08-25, see above; the ON arms are still seed-0 only).
- Gate HalfHop to train-only (currently active at eval too) to separate its train vs eval contribution.
