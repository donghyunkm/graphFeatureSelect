# GR2ST-style GATv2 attention-entropy regularization

_Added 2026-09-02. Implemented + unit-tested; λ ∈ {0.0, 0.001, 0.01, 0.05} sweep **COMPLETE** as of 2026-09-05 (see Results)._

## Goal
Encourage **sharper GATv2 edge-attention**: for a given destination node/head, a small subset of
spatial neighbors should carry most of the attention mass instead of it being diffuse. We port the
auxiliary attention-entropy loss from GR2ST (Zhang et al.) into Antelope's Lightning pipeline.

## What was added
- **`gfs/configs/antelope.yaml`** (`model:` section, repo defaults keep behavior unchanged):
  - `attention_entropy_weight: 0.0` — λ on the entropy loss. `0.0` disables it (default).
  - `attention_entropy_eps: 1.0e-9` — the GR2ST `log(alpha + eps)` floor.
  - `log_attention_stats: false` — **decouples measuring from regularizing**. When `true` (GAT only), the
    entropy + neighbourhood diagnostics are computed and logged even with `attention_entropy_weight=0`, so a
    baseline run's attention stats are directly comparable to a regularized run's. It does NOT add anything
    to the loss. Numerically free (requesting attention weights gives bit-identical logits); off by default.
  - Backward-compatible: `LitGnnFs` reads them with `.get(..., default)`, so the other `antelope*.yaml`
    configs (which lack these keys) are unaffected.
- **`GnnFs.forward(..., return_attention_weights=False)`** — new optional flag. When `False` (default)
  the forward path is byte-for-byte the previous implementation and returns just the logits. When `True`
  (GAT only) it also returns `attention_records = [(att_edge_index, alpha), ...]`, one entry per GAT
  layer, taken directly from `GATv2Conv(..., return_attention_weights=True)`. Records are per-batch, not
  persistent module state. `GnnFs.gnn_type` is now stored.
- **`LitGnnFs`**:
  - `__init__` stores `attention_entropy_weight`, `attention_entropy_eps`, `gnn_type`, and raises
    `ValueError` if `attention_entropy_weight > 0` with `gnn != "gat"`.
  - `forward(..., return_attention_weights=False)` mirrors the flag to `self.model`.
  - `_gr2st_attention_entropy(alpha)` — the **literal** GR2ST loss
    `-mean_e[ sum_h alpha_{e,h} log(alpha_{e,h}+eps) ]`, `alpha` shaped `[E, H]`. Heads are NOT averaged.
  - `_attention_sharpness_diagnostics(records)` — detached, per-destination-node/head diagnostics:
    top-1 mass, effective neighbours `exp(H)`, and degree-normalized entropy `H/log(deg)` (deg>1 only).
  - `training_step` composes `train_loss_total = classification + scGist_reg + λ·attention_entropy`
    and **returns `train_loss_total`** (Lightning automatic optimization handles backward/step; no manual
    `.backward()` / `optimizer.step()`). Two independent gates: `use_attention_entropy` (weight>0) adds the
    loss term; `collect_attention` (weight>0 **or** `log_attention_stats`) gathers attention to log. When
    collecting for logging only (weight=0), the entropy is computed under `torch.no_grad()` and never added.
  - `validation_step` optionally logs `val_attention_entropy` + val diagnostics **as diagnostics only** —
    never added to `val_loss_ce`, never used for checkpoint selection.

## Key design decisions (adaptations of GR2ST to Antelope)
- **Loss = literal GR2ST form.** Not normalized/per-node entropy, KL, top-k, sparsemax/entmax — those are
  separate future ablations. The normalized-entropy quantity appears only as a diagnostic.
- **Stacked GAT layers → MEAN across layers**, not sum. GR2ST sums the entropies of its two single-GAT
  branches; Antelope has `local_layers` sequential GAT layers, so summing would make λ's effective
  strength scale with depth. Mean keeps λ comparable across `local_layers` settings.
- **Entropy over ALL GAT edges in the forward pass**, including HalfHop slow-node edges and (under
  `trainmode=1/2`) context edges — even though the classification loss is restricted to a node subset.
  Those edges shape the root representations, so this is the closest analogue to regularizing the GAT
  itself. Computed on the post-transform `data.edge_index` (the exact graph used for that batch's
  predictions), not a separate pass on the raw graph.
- **`tau` is untouched.** The Gumbel-softmax feature-selection temperature is unrelated to attention
  sharpness; `tau_schedule` was not modified and no GAT softmax temperature was introduced.
- **Checkpointing unchanged.** `gfs/trainers/antelope.py` still monitors `val_f1_overall` (max). We want
  the best classifier, not the lowest-entropy model. The trainer required **no** loss-related changes.

## PyG normalization (verified empirically, PyG 2.8.0 / torch 2.6.0)
`GATv2Conv(..., return_attention_weights=True)` returns `alpha` of shape `[E, H]`, normalized so that for
each **destination** node and head the incoming attention sums to 1. The destination is **row 1** of the
returned `att_edge_index` (flow `source_to_target`). Confirmed on a tiny graph; asserted in the tests.

## Logging (TensorBoard via the existing `self.log`)
- Training: `train_loss_cls`, `train_loss_total`, `train_loss_ce` (backward-compat alias =
  classification + scGist reg), `train_loss_reg` (scGist only), and — when λ>0 —
  `train_attention_entropy`, `train_loss_attention_entropy_weighted`,
  `train_attention_top1_mass`, `train_attention_effective_neighbors`, `train_attention_normalized_entropy`.
- Validation (λ>0, diagnostics only): `val_attention_entropy` + the three `val_attention_*` sharpness
  metrics. `val_f1_overall`/`val_macro_acc`/`val_overall_acc`/`val_loss_ce` unchanged.

## Tests — `tests/test_attention_entropy.py` (13 tests, all pass on CPU)
Backward-compat (disabled path matches old logits, default returns only logits); literal entropy helper
(finite, ≥0, sharp<diffuse); gradients reach GATv2 `att` params (finite, ≥1 nonzero); multi-head `alpha`
stays 2-D; multi-layer mean aggregation; PyG per-destination/head normalization ≈1; non-GAT builds at λ=0
and raises at λ>0 for gcn/sage; sharpness-diagnostic ranges; `training_step` total-loss composition; and
two tiny end-to-end Lightning fits (λ>0 logs entropy metrics + `val_f1_overall`; λ=0 takes the plain path).
`pytest` was installed into the `gfs` env to run them (not yet added to `pyproject.toml` dev deps).

## How to run the intended experiment (user must `sbatch` / launch)
Hydra override, everything else identical between arms:
```bash
# baseline — log attention stats WITHOUT regularizing, so it's comparable to the regularized arm
python -u ../gfs/trainers/antelope.py model.attention_entropy_weight=0.0 model.log_attention_stats=true data.prefix=entropy_lam0
# regularized (GR2ST default) — diagnostics are logged automatically when weight>0
python -u ../gfs/trainers/antelope.py model.attention_entropy_weight=0.01 data.prefix=entropy_lam0p01
```
Set `model.log_attention_stats=true` on the baseline arm so both runs log `train_attention_entropy`,
`train_attention_top1_mass`, `train_attention_effective_neighbors`, `train_attention_normalized_entropy`
(+ `val_*` variants) — letting you compare the attention distributions head-to-head.
Suggested sweep: `0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1` (min: `0, 0.001, 0.01, 0.1`).

**Success criterion (not just "entropy went down"):** attention entropy ↓, normalized entropy ↓, top-1
mass ↑, effective neighbours ↓ **while** `val_f1_overall` / `val_macro_acc` hold or improve. If attention
collapses and prediction drops, λ is too strong. Recompute test F1 from `test_pred.pt`, not plots.

## First comparison run — `scripts/main_4hop_gr2st_compare.sh` (ready to launch)
A paired on/off comparison, added 2026-09-02: two **4-hop, 100-epoch, select-5, raw-graph (HalfHop OFF)**
runs on the full 22-label data (trainer `antelope_4hop.py`, config `antelope_4hop.yaml`), Slurm `--array=0-1`:
- **task 0 (baseline):** `attention_entropy_weight=0.0`, `log_attention_stats=true` — logs the entropy +
  neighbourhood diagnostics without applying the loss.
- **task 1 (GR2ST):** `attention_entropy_weight=0.01` (GR2ST default) — diagnostics logged automatically.

Both use **the same `rand_seed=0`** (identical data split + init), so the *only* difference is the entropy
loss → a clean paired test of its effect. `model.log_attention_stats=true` on both so `train/val`
`attention_entropy`, `top1_mass`, `effective_neighbors`, `normalized_entropy` are directly comparable.
Runtime ~13–28h/seed (100ep, well under the 60h cap). Compare per the success criterion above; recompute
test F1 from `test_pred.pt`.

**Weight choice — why 0.01.** The entropy loss magnitude here is O(0.5–1) at init (roughly `H·log(deg)/deg`
for near-uniform attention, H=3 heads), vs classification CE ~log(22)≈3.1. So λ=0.01 adds ~0.5–1% of the CE
at start — the gentle GR2ST default that sharpens attention without dominating the objective. If the on/off
run shows entropy/top-1-mass barely move while F1 holds, step λ up (0.03 → 0.1); if attention collapses and
F1 drops, λ is too strong. A fuller sweep (`0, 1e-3, 1e-2, 3e-2, 1e-1`) is the natural follow-up.

## Status — λ sweep COMPLETE (as of 2026-09-05)
Implemented + unit-tested. The on/off pair **launched 2026-09-02** (`main_4hop_gr2st_compare.sh`, array
**25553166**); two more weight points **launched 2026-09-03** (`main_4hop_gr2st_ae0p001.sh`,
`main_4hop_gr2st_ae0p05.sh`). All four arms are **seed 0, 4-hop / 100ep / select-5 / HalfHop-OFF, full
22-label data, identical splits** — the only difference is λ. Sweep now λ ∈ {0.0, 0.001, 0.01, 0.05}.

| λ | job | prefix | log dir (`data/logs/`) | script | status |
|---|---|---|---|---|---|
| 0.0 (baseline) | 25553166_0 | `full_4hop100_ae0_nohh` | `20260902_222303_full_4hop100_ae0_nohh_s0` | `main_4hop_gr2st_compare.sh` (task 0) | ✅ done |
| 0.001 | 25555463 | `full_4hop100_ae0p001_nohh` | `20260903_065929_full_4hop100_ae0p001_nohh_s0` | `main_4hop_gr2st_ae0p001.sh` | ✅ done |
| 0.01 (GR2ST default) | 25553166_1 | `full_4hop100_ae0p01_nohh` | `20260902_222232_full_4hop100_ae0p01_nohh_s0` | `main_4hop_gr2st_compare.sh` (task 1) | ✅ done |
| 0.05 | 25555510 | `full_4hop100_ae0p05_nohh` | `20260903_071538_full_4hop100_ae0p05_nohh_s0` | `main_4hop_gr2st_ae0p05.sh` | ✅ done |

Logs are **TensorBoard event files only** (no `metrics.csv`); read scalars with
`tensorboard.backend.event_processing.event_accumulator.EventAccumulator`. Timing on 25553166_1: ~17.3
min/epoch → ~28.5 h wall to ep100, well under the 60 h `--time` cap; the other 4-hop arms run at the same rate.

### Mid-run snapshot @ ep30 — UNVERIFIED, mid-anneal (do NOT conclude from this)
The 0.01 arm sharpens attention strongly and the loss term stays tiny (`train_loss_attention_entropy_weighted`
≈ 0.002), confirming the mechanism half of the hypothesis. `val_f1` is still ~0.08–0.12 for all arms because
tau is mid-anneal (≈1.45, not yet committed) — this is expected undertraining, **not** a result; wait for ep100.

| λ | val_attention_entropy | val_effective_neighbors | val_top1_mass |
|---|---|---|---|
| 0.0 | 0.359 | 14.2 | 0.134 |
| 0.01 | 0.246 | 6.9 | 0.334 |

Panels @ ep25–29 are **identical across the 0.0 and 0.01 arms**: gene indices `{324, 105, 472, 280, 228}`
(`selections.csv` per arm; these are indices 0–499, not gene names). So far the entropy loss sharpens
attention without changing *which* genes are selected — but this is mid-anneal and can still shift.

**On completion (all four arms):** recompute test F1 from each `test_pred.pt`; tabulate `train/val`
`attention_entropy`, `top1_mass`, `effective_neighbors`, `normalized_entropy` vs λ against the success
criterion above (sharper attention WHILE `val_f1` holds). The λ=0.0 baseline should track the seed-0 4-hop
100ep result from `docs/4hop_2hop_comparison.md` (0.6849 wtd / 0.5056 macro), up to rerun nondeterminism.
Note `selections.csv` `epoch` runs one ahead of the TensorBoard `epoch` scalar (off-by-one index; same run).

## Results — full λ sweep complete (2026-09-05)
All four arms finished 100 epochs (tested from best-val checkpoint). Test F1 recomputed from `test_pred.pt`
(n=31,562); final-epoch (ep99) val attention diagnostics from the training event file.

| λ | test wtd F1 | test macro F1 | final panel (gene idx) | val eff-neighbors | val entropy | val top1-mass |
|---|---|---|---|---|---|---|
| 0.0 (baseline) | 0.6849 | 0.5056 | {2, 83, 225, 250, 480} | 16.9 | 0.400 | 0.122 |
| 0.001 | **0.7047** | **0.5378** | {2, 12, 83, 225, 480} | 16.7 | 0.398 | 0.126 |
| 0.01 (GR2ST default) | **0.7052** | 0.5309 | {2, 12, 83, 225, 480} | 15.5 | 0.387 | 0.140 |
| 0.05 | 0.6375 | 0.4624 | {2, 83, 435, 480} | 5.5 | 0.147 | 0.669 |

**Findings.**
- **Baseline sanity check passes** — λ=0.0 lands exactly on the seed-0 4-hop 100ep reference (0.6849 wtd / 0.5056 macro).
- **Gentle entropy pressure helps.** λ=0.001 and λ=0.01 both improve F1 (+0.020 wtd over baseline) and both
  recover the stable seed-0 panel {2, 12, 83, 225, 480} — gene 12 comes back that plain 4-hop seed-0 dropped.
  The two are essentially tied on weighted F1; λ=0.001 edges macro slightly higher (0.5378 vs 0.5309). At these
  weights the persistent attention sharpening at ep99 is modest (eff-neighbors 16.9→16.7→15.5) — the tau anneal
  re-diffuses attention as selection commits, so the strong ep30 mid-anneal sharpening was NOT the final state.
- **Over-sharpening costs F1 — bracketed at λ=0.05.** λ=0.05 collapses attention hard (eff-neighbors 16.9→5.5,
  top1-mass 0.12→0.67, entropy 0.40→0.15) and weighted F1 drops to 0.6375 — *below* the λ=0 baseline. The panel
  also drifts off the backbone (drops 12 and 225, picks up 435). This answers where over-sharpening starts to
  hurt: the success criterion (attention sharpens WHILE val_f1 holds) fails at 0.05, so **no need to extend to 0.1**.
- **Sweep verdict:** the useful λ window is ~0.001–0.01 (gentle). Below it the loss is inert; at 0.05 it
  dominates and degrades both the classifier and the panel. Recommend λ=0.01 (GR2ST default) as the operating point.

### Panel comparison vs the 2-hop model (seed 0, 100ep)
The 2-hop seed-0 panel (100ep, `docs/4hop_2hop_comparison.md`) is **{2, 12, 83, 225, 480}**.

| model | panel (gene idx) | vs 2-hop seed 0 |
|---|---|---|
| 2-hop seed 0 | {2, 12, 83, 225, 480} | — |
| 4-hop λ=0.0 (plain) | {2, 83, 225, 250, 480} | drops 12, swaps 225→250 (differs by 1 gene) |
| 4-hop λ=0.001 | {2, 12, 83, 225, 480} | **identical** |
| 4-hop λ=0.01 | {2, 12, 83, 225, 480} | **identical** |
| 4-hop λ=0.05 | {2, 83, 435, 480} | drops 12 & 225, adds 435 (differs by 2 genes) |

- **The gentle-entropy 4-hop arms (λ=0.001, 0.01) recover the exact 2-hop seed-0 panel.** Plain 4-hop seed-0
  (λ=0) is the outlier — it uniquely drops gene 12 and picks up 250 instead. Attention-entropy pressure nudges
  the noisy 4-hop seed-0 selection back onto gene 12 and the stable {2,12,83,225,480} panel, which is why its
  F1 rises to ~0.705, matching the 2-hop result.
- **Over-sharpening (λ=0.05) breaks the panel** — it drifts off the {12, 225} backbone (adds 435), and F1 falls
  to 0.6375. The {2, 83, 480} core is shared by every arm except the drift in 0.05.
- **Caveat:** seed 0 only — "recovers the 2-hop panel" is a single-seed observation; multi-seed error bars on
  λ=0.01 vs 0.0 would be needed to lean on the +0.020 F1 gain and the panel-recovery claim.

### Mid-anneal snapshot vs final — confirmed NOT a result
At ep30 the 0.01 arm's val eff-neighbors were 6.9 (vs 14.2 baseline); by ep99 they relaxed to 15.5 vs 16.9.
The ep30 table earlier in this doc is confirmed a mid-tau-anneal transient, not the converged attention state.
