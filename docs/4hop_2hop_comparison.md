# 4-hop vs 2-hop on the full 22-label data

**Status:** ✅ **COMPLETE** (2026-09-01). Both arms finished all 5 seeds; F1 recomputed from
`test_pred.pt` and verified against the logged `test_f1_overall` (exact match). **Answer: 4-hop does NOT
beat 2-hop on the full 22-label data** — the two arms are statistically tied (4-hop 0.7099 ± 0.0142
weighted vs 2-hop 0.7049 ± 0.0070; 4-hop is slightly *worse* on macro), so the abundant-subclass
+0.0132 w / +0.0181 m win **fails to replicate** at full label resolution.

The mid-run "divergence" alarm (below, from a 2026-08-31 06:20 snapshot) was a **false alarm — transient
mid-anneal state, not pathology.** At that snapshot seeds 1–4 were only at epoch ~20 with `tau` still 2.5
(rising, not yet past its schedule's turning point) and selection uncommitted, so val_f1 was low and
val_loss high. By epoch 99 `tau` had annealed and selection committed (probs ≈ 0.999), val recovered, and
final `test_loss_ce` is a healthy 0.85–0.94 (not the mid-run 11–15). Seed 0 (the slow one, ~12 h) is a
mild low outlier (0.6849) but non-degenerate. No bug; the 4-hop path was just undertrained at the snapshot.

## Question
Does complete-receptive-field training (4-hop sampling + shallow/complete-field loss) beat the
standard 2-hop baseline on the **full** isocortex dataset (500 genes, 22 subclasses)? This replicates
the abundant-subclass finding (`docs/20260820_deep_sample_shallow_loss.md`: +0.0132 weighted /
+0.0181 macro F1, 4-hop > 2-hop on all 5 seeds) at full label resolution.

## Design
Two arms, seeds 0–4 (5 each → 10 runs). GFS with selection (`fs_method=persist`, `n_select=5`).

| Arm | trainer | config | sampling | training loss |
|---|---|---|---|---|
| **2-hop (baseline)** | `antelope.py` | `antelope.yaml` | `n_hops=2` | `trainmode=0`: all patch nodes (mostly incomplete 2-hop fields) |
| **4-hop (complete field)** | `antelope_4hop.py` | `antelope_4hop.yaml` | `n_hops=4`, `loss_hops=2` | `trainmode=2`: only nodes within 2 hops of a seed (complete field, via `batch.loss_mask`) |

**Held identical across arms** (only sampling depth + which nodes contribute to loss differ):
`n_labels=22`, `n_genes=500`, `n_select=5`, `fs_method=persist`, `batch_size=64`, `local_layers=2`
(message passing is 2 hops in **both**), `xyz_status=true`, `tautype=old`, 100 epochs, lr=1e-3
constant. **HalfHop OFF** for train *and* eval in both arms (raw-graph eval; avoids the eval-time
HalfHop perturbation gotcha — see `MEMORY.md` `eval-time-halfhop-active`). Val/test scored on
**root/seed nodes only** in both arms (`antelope.py:487,578`).

The only differences between arms are exactly: sampling depth (`n_hops` 2 vs 4) and the loss-node set
(`trainmode` 0 vs 2). This is the same diagonal as the abundant 2×2, moved to full data.

### What the two arms actually differ on (the loss node set is the *same*)

The key point: **both arms take the training loss on the same set of nodes — every node within 2 hops
of a seed. The only difference is whether those nodes have a complete receptive field.**

- **2-hop (`trainmode=0`):** `NeighborLoader(num_neighbors=[-1,-1])` samples every node within 2 hops of
  the batch's 64 seeds; loss is on `train_mask` = **all** of them (seeds + hop-1 + hop-2). But under
  2-hop sampling only the **seed** has a complete 2-hop field. A hop-1 loss node's 2-layer output needs
  up to hop-3 neighbors, a hop-2 node needs up to hop-4 — none of which were sampled. So every non-seed
  loss node is trained on a **truncated** neighborhood → wrong embedding.
- **4-hop (`trainmode=2`):** `NeighborLoader(num_neighbors=[-1]*4)` samples every node within 4 hops of
  the same seeds; `loss_mask` restricts the loss to nodes within **2 hops** of a seed — i.e. **the same
  within-2-hop set**. Now those hop-1/hop-2 loss nodes have their full 2-hop closure materialised (the
  extra hops 3–4 were sampled precisely to complete their neighborhoods), so every loss node has a
  **correct** embedding. Hops 3–4 carry no loss and message passing is still 2 layers.

Because `num_neighbors=[-1]` is **exhaustive** (all neighbors, not a random subsample), the within-2-hop
set is deterministic given the seeds, so the two arms' loss-node sets genuinely coincide (not merely
match in size). The manipulation is purely *incomplete-field vs complete-field embeddings on the same
nodes* — confirmed by the matched loss-node volume in the abundant arm (~4,650 4-hop vs ~4,147 2-hop).
Message passing is 2 hops and eval is root-only in both arms; nothing else changes.

## How to run
```bash
cd /home/donghyun.kim/graphFeatureSelect/scripts
sbatch main_fulldata_2hop_seeds.sh   # array 0-4, 2-hop baseline
sbatch main_fulldata_4hop_seeds.sh   # array 0-4, 4-hop
```
Log dirs: `data/logs/<datetime>_full_{2hop,4hop}_seed{0-4}_nohh_s{0-4}/`.
Runtime caveat: full data is much larger than the abundant subset, so 4-hop patches are large and the
4-hop arm is expected to be substantially slower than 2-hop.

## Runs / Slurm IDs
Launched 2026-08-30. 2-hop array = **25507587**, 4-hop array = **25507593** (both `--array=0-4`,
array task id → seed). Log dirs / statuses to confirm on completion.

| Arm | seed | Slurm job | log dir | status |
|---|---|---|---|---|
| 2-hop | 0 | 25507587_0 | `data/logs/20260830_180252_full_2hop_seed0_nohh_s0` | ✅ done (100 ep) |
| 2-hop | 1 | 25507587_1 | `data/logs/20260830_180246_full_2hop_seed1_nohh_s1` | ✅ done (100 ep) |
| 2-hop | 2 | 25507587_2 | `data/logs/20260830_180248_full_2hop_seed2_nohh_s2` | ✅ done (100 ep) |
| 2-hop | 3 | 25507587_3 | `data/logs/20260830_180250_full_2hop_seed3_nohh_s3` | ✅ done (100 ep) |
| 2-hop | 4 | 25507587_4 | `data/logs/20260830_180246_full_2hop_seed4_nohh_s4` | ✅ done (100 ep) |
| 4-hop | 0 | 25507593_0 | `data/logs/20260830_180246_full_4hop_seed0_nohh_s0` | ✅ done (100 ep) |
| 4-hop | 1 | 25507593_1 | `data/logs/20260831_000225_full_4hop_seed1_nohh_s1` | ✅ done (100 ep) |
| 4-hop | 2 | 25507593_2 | `data/logs/20260831_001837_full_4hop_seed2_nohh_s2` | ✅ done (100 ep) |
| 4-hop | 3 | 25507593_3 | `data/logs/20260831_002142_full_4hop_seed3_nohh_s3` | ✅ done (100 ep) |
| 4-hop | 4 | 25507593_4 | `data/logs/20260831_003110_full_4hop_seed4_nohh_s4` | ✅ done (100 ep) |

**Historical note — the mid-run "divergence" was a false alarm.** A 2026-08-31 06:20 TensorBoard snapshot
(taken while seeds 1–4 were only ~ep20) showed train_acc ≈ 0.95 with val_f1 ≈ 0.05–0.15, val_loss 11–15,
and `tau` at 2.5–2.7 (rising), which looked like a severe generalization failure or a bug in the 4-hop
path. It was neither: the runs were simply **undertrained and mid-tau-anneal** — `tau` had not yet passed
its schedule's turning point, so selection was uncommitted. Left to finish all 100 epochs, `tau` annealed,
selection hardened (final probs ≈ 0.999), val recovered, and final `test_loss_ce` is 0.85–0.94. Lesson:
don't diagnose the 4-hop arm from an early-epoch snapshot — it anneals late. (The `.out`/`.err` still only
show the model summary; Lightning's progress bar isn't flushed — read tfevents, and wait for completion.)

## Results
Weighted + macro F1 recomputed from `test_pred.pt` (n = 31,562 test cells) with
`sklearn.metrics.f1_score(average="weighted"|"macro")`. Recomputed weighted F1 matches the logged
`test_f1_overall` exactly (so "overall" = weighted here) — for **both** arms.

Per seed:

| seed | 2-hop weighted | 2-hop macro | 4-hop weighted | 4-hop macro |
|---|---|---|---|---|
| 0 | 0.7019 | 0.5341 | 0.6849 | 0.5056 |
| 1 | 0.7164 | 0.5359 | 0.7185 | 0.5318 |
| 2 | 0.6993 | 0.5292 | 0.7145 | 0.5246 |
| 3 | 0.7063 | 0.5429 | 0.7190 | 0.5297 |
| 4 | 0.7004 | 0.5276 | 0.7125 | 0.5442 |

Summary (mean ± sd over seeds 0–4):

| Arm | weighted (mean ± sd) | macro (mean ± sd) |
|---|---|---|
| 2-hop / tm0 | 0.7049 ± 0.0070 | **0.5339 ± 0.0061** |
| 4-hop / tm2 | **0.7099 ± 0.0142** | 0.5272 ± 0.0141 |

**4-hop − 2-hop = +0.0050 weighted, −0.0067 macro.** Both differences are smaller than the (roughly
doubled) 4-hop cross-seed spread, so the arms are statistically **tied**. 4-hop is also markedly noisier
(sd ~0.014 vs ~0.007) and its slow seed 0 is a low outlier (0.6849).

Paired-by-seed significance (n=5; paired t = primary + Cohen dz + sign, per `progress.md` convention).
Bootstrap redone 2026-09-25 = **cell-level paired bootstrap**, B=10 000 (`gfs/analysis/paired_cell_bootstrap.py`):
one shared test-cell resample (with replacement) applied across all 5 seeds × both arms, per-seed F1
differenced within seed then averaged over the 5 seeds; 95% percentile CI + 2-sided p. ⚠️ **Scope:
cell-level uncertainty within one biological sample — NOT across-sample/donor generalization**; it is far
higher-powered than the seed-level paired t-test and can disagree with it. Supersedes the earlier seed-level
bootstrap (B=50 000).

| metric | Δ (4−2) | sign | paired t p | bootstrap p (95% CI) | Cohen dz |
|---|---|---|---|---|---|
| weighted | +0.0050 | 4/5 | 0.45 (ns) | <0.0002 [+0.0030, +0.0071] | +0.38 |
| macro | −0.0067 | 1/5 | 0.41 (ns) | <0.0002 [−0.0099, −0.0036] | −0.41 |

**The paired t-test (primary) calls both metrics ns — so the tie stands as the headline cross-seed
generalization conclusion**, and 4-hop does **not** replicate its abundant-subset win at full 22-label
resolution (memory `4hop-does-not-beat-2hop-full-data`). The higher-powered cell-level bootstrap, however,
now flags **both** metrics as significant *on this sample*: weighted significantly positive (CI entirely
above zero) and **macro significantly negative** (CI entirely below zero — i.e. 4-hop is slightly *worse* on
macro). This is the expected power/scope gap, not a contradiction: the bootstrap measures cell-level
uncertainty within a single biological sample (tiny CIs because ~31.5k cells are shared across seeds),
whereas the t-test measures whether the effect generalizes across seeds — and the small, sign-inconsistent
per-seed Δ (weighted 4/5 but tiny, macro 1/5 and negative) does not. The tie conclusion rests on the t-test.

Selected panels per seed (gene indices, final epoch 99; all with selection prob ≈ 0.99, fully committed):

| seed | 2-hop panel | 4-hop panel |
|---|---|---|
| 0 | 2, 12, 83, 225, 480 | 2, 83, 225, 250, 480 |
| 1 | 2, 12, 83, 435, 480 | 2, 12, 83, 435, 480 |
| 2 | 2, 12, 83, 225, 480 | 2, 12, 83, 435, 480 |
| 3 | 2, 12, 83, 207, 480 | 2, 12, 83, 435, 480 |
| 4 | 2, 12, 83, 225, 480 | 2, 12, 83, 271, 480 |

**Panel stability:** 2-hop core **{2, 12, 83, 480}** in all 5 seeds (5th gene: 225 ×3, 207/435 ×1).
4-hop core **{2, 83, 480}** in all 5, with **12** in 4/5 (dropped only in seed 0, which took 225+250
instead) and **435** in 3/5. Both arms converge on the same backbone; the panels are effectively the same
selection, consistent with selection being graph-invariant.

## Follow-up: 300-epoch probe (longer tau exploration)

**Motivation.** The 100ep tie is *not* an undertraining artifact — both arms are fully converged and
plateaued at ep99 (val_f1 flat over the last ~12 epochs; `tau` annealed to ~0.01). But the `tautype="old"`
schedule `tau = 10·(0.01/10)^(epoch/max_epochs)` **stretches with `max_epochs`**, so training longer gives
the selector more gradient steps in the high-tau *exploratory* regime before it commits, and a longer
committed phase to fit the classifier on the chosen panel. Hypothesis: more exploration might discover a
different/better 5-gene panel and let 4-hop pull ahead (or move both arms).

**Design.** Identical to the 100ep arms (select-5, HalfHop off train+eval, full 22-label, `n_hops` 2/4,
`trainmode` 0/2) except `trainer.max_epochs=300`. Probe scale: **2 seeds each** (array 0-1). At 300ep the
schedule reaches tau<1.0 at ep101 and tau<0.1 (near-committed) at ep201 (vs ep34 / ep67 at 100ep) — ~3× the
exploratory window. Scripts: `main_fulldata_2hop_300ep_seeds.sh`, `main_fulldata_4hop_300ep_seeds.sh`.

**Runs (launched 2026-09-01).** 2-hop 300ep = job **25522891**, 4-hop 300ep = job **25522893** (both
`--array=0-1`, task id → seed). Prefixes `full_{2hop,4hop}300_seed{0,1}_nohh`. 4-hop ~36h/seed (within the
60h walltime).
**Caveat:** these are directly comparable to the 100ep *full-data* arms (also HalfHop-off, select-5), NOT to
the abundant 100ep win — an abundant-vs-full test at 300ep would require rerunning the abundant arms at 300ep.

**Progress / partial results (as of 2026-09-02).** Both **2-hop** seeds have finished; both **4-hop** seeds
are still mid-anneal.

| Arm | seed | log dir | status | weighted F1 | macro F1 | panel (best-val-F1 ckpt) |
|---|---|---|---|---|---|---|
| 2-hop 300ep | 0 | `20260901_074102_full_2hop300_seed0_nohh_s0` | ✅ done (300 ep) | **0.7006** | 0.5054 | 2, 12, 83, 435, 480 |
| 2-hop 300ep | 1 | `20260902_023141_full_2hop300_seed1_nohh_s1` | ✅ done (300 ep) | **0.7114** | 0.5249 | 2, 12, 83, 435, 480 |
| 4-hop 300ep | 0 | `20260902_023141_full_4hop300_seed0_nohh_s0` | ⛔ CANCELLED (~ep68/300, would time out) | — | — | uncommitted (mid-anneal) |
| 4-hop 300ep | 1 | `20260902_081935_full_4hop300_seed1_nohh_s1` | ⛔ CANCELLED (~ep48/300, would time out) | — | — | uncommitted (mid-anneal) |

F1 recomputed from `test_pred.pt` (n = 31,562; argmax over 22-class logits), matches logged
`test_f1_overall` exactly for both finished seeds. Test runs on the **best-`val_f1_overall`** checkpoint
(`ckpt_path="best"`, `mode="max"`), i.e. the best-val-F1 epoch (~264 / ~251), not the final epoch — but
selection committed early and is identical `{2,12,83,435,480}` at the best-F1, best-val-loss, and final
epochs, so the panel is unaffected. **Read (2-hop, both seeds done):** more exploration did
**not** help — 2-hop 300ep = 0.7006 / 0.7114 w (mean ≈ 0.706), statistically the same as the 100ep 2-hop
mean (0.7049 w). **Both** seeds committed to the identical panel {2, 12, 83, 435, 480} (probs ≈ 0.999) —
the {2,12,83,480} core plus 435 (already seen in the 100ep seed 1). No movement off the core from the ~3×
longer exploratory window.

**⏸️ 4-hop 300ep probe CANCELLED 2026-09-02 — TO RESUME LATER.** Both 4-hop 300ep seeds (job
`25522893_0/_1`) were cancelled before finishing. The sbatch script's runtime estimate ("~36h/seed") was
wrong: measured cost is **~17 min/epoch → ~85h for 300 epochs**, far over the 60h `--time` cap. They'd have
been SIGKILLed at ~ep211/215 — well short of 300 and, worse, **mid-tau-anneal** (the `"old"` schedule ties
tau to `max_epochs`, so at ep211 tau ≈ 0.078, selection uncommitted), so the partial checkpoints are
unusable, not merely "shorter." No usable results from this probe.

**How to resume (not yet implemented):** the chosen path is **checkpoint-resume** — keep a per-run
checkpoint and requeue across multiple 60h blocks so the full 300-epoch anneal completes without needing a
single >60h wall-clock allocation (celltypes QOS likely caps `--time` below the ~90h a single-shot run
would need). This requires verifying/adding resume-from-checkpoint support in `antelope_4hop.py` (Lightning
`ckpt_path=` on `trainer.fit`, plus Slurm `--requeue`/auto-resubmit). Deferred; the 100ep result stands as
the answer to the 4-hop-vs-2-hop question — this probe only tests the longer-exploration hypothesis and is
lower priority.

## Cross-reference: attention-entropy regularization recovers 2-hop-level seed 0 (`docs/20260902_gr2st_attention_entropy.md`)
The GR2ST attention-entropy sweep (seed 0, 4-hop / 100ep / select-5 / HalfHop-OFF — same config as this
doc's 4-hop arm) gives a clean read against **seed 0** here. Comparing to seed-0 rows above:

| model | weighted F1 | macro F1 | panel (gene idx) |
|---|---|---|---|
| 2-hop seed 0 (this doc) | 0.7019 | 0.5341 | 2, 12, 83, 225, 480 |
| 4-hop seed 0 (this doc) | 0.6849 | 0.5056 | 2, 83, 225, 250, 480 |
| 4-hop seed 0 + attn-entropy λ=0.0 | 0.6849 | 0.5056 | 2, 83, 225, 250, 480 |
| 4-hop seed 0 + attn-entropy λ=0.01 | **0.7052** | 0.5309 | **2, 12, 83, 225, 480** |

- The λ=0.0 arm **exactly reproduces** the 4-hop seed-0 run here (identical F1 *and* panel) — a bit-for-bit
  sanity check that the entropy machinery is inert at λ=0 and the seed-0 4-hop deficit is real, not noise.
- **λ=0.01 closes the 4-hop seed-0 deficit and lands on the 2-hop solution.** Its panel is *identical* to
  2-hop seed 0's `{2, 12, 83, 225, 480}` — it recovers gene **12** (the {2,12,83,480} backbone that plain
  4-hop seed 0 uniquely dropped for 225+250) — and its weighted F1 (0.7052) matches/slightly beats 2-hop
  seed 0 (0.7019). So sharpening attention nudged the noisy 4-hop seed 0 back onto the stable panel the
  other 4-hop seeds and the 2-hop arm already find. Whether this holds across seeds is untested (seed 0 only).

## Findings
- **4-hop does NOT beat 2-hop on the full 22-label data.** 4-hop 0.7099 ± 0.0142 weighted / 0.5272 ± 0.0141
  macro vs 2-hop 0.7049 ± 0.0070 / 0.5339 ± 0.0061 — a +0.005 weighted / −0.007 macro difference well
  inside the noise. **The abundant-subclass +0.0132 w / +0.0181 m win fails to replicate at full label
  resolution.** Likely the complete-receptive-field benefit is specific to the easier 7-label subset (or is
  swamped by the harder 22-way problem); 4-hop also carries ~2× the cross-seed variance for no gain.
- **2-hop baseline** at full label resolution (22 subclasses, 500 genes, n_select=5): weighted F1
  **0.7049 ± 0.0070**, macro **0.5339 ± 0.0061** over seeds 0–4. Tight cross-seed spread.
- **The mid-run 4-hop "divergence" was a false alarm** — transient undertrained/mid-tau-anneal state, not a
  bug. All 5 seeds finished cleanly (recomputed F1 = logged; test_loss_ce 0.85–0.94; selection committed to
  probs ≈ 0.999). Don't diagnose this arm from an early-epoch snapshot.
- Panels stable and shared across both arms (core {2, 83, 480} + 12); reproducible selection.
