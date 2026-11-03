# graphFeatureSelect (`gfs`)

Gene panel selection for spatial transcriptomics. Learns which **~10–20 genes out of 500**
form the most informative panel for cell-type classification, by training a GNN whose input
passes through a differentiable feature-selection layer, then reading off the selections at
convergence.

Motivation: post-hoc transcriptomic profiling of the same brain tissue used for
activity/morphology imaging can only measure a small number of genes, so the panel choice
matters.

## Layout

```
gfs/
  configs/       hydra yaml configs (antelope.yaml is the active one)
  data/          anndata -> PyG conversion + lightning datamodules
                 node_exposure.py: count per-cell loss terms/epoch, build upsampled tabular set
                 panel_tabular.py: serve that set as self-edge PyG batches (frozen-panel baseline)
                 hemisphere_new_panel.py: frozen panel WITH the graph kept (noise-robustness arm)
  models/        model variants, one file per feature-selection method
    samplers/    differentiable top-k sampler zoo (SIMPLE, I-MLE, Gumbel, SFESS, PPS)
    stg/         stochastic-gates (L0) layers
  trainers/      one hydra entrypoint per model variant
scripts/         SLURM sbatch wrappers, one per variant
docs/            does-the-graph-help.md: standalone argument on whether neighbor info helps
notebooks/       exploratory work, roughly chronological by filename date
topk_*.py        standalone synthetic sanity checks (make_classification, no graph)
data/            h5ad inputs, logs/, checkpoints/ (gitignored)
outs/            SLURM stdout/stderr
```

`config.toml` holds `data_root`; read via `gfs.utils.get_paths()`, which resolves it
relative to the package root.

## Pipeline

**Preprocess** (`gfs/data/preprocess.py`) — raw `.h5ad` (Allen ABC-style MERFISH) → drop
cell types with <5 cells → kNN graph (k=20) on `(x_section, y_section)` stored in
`adata.obsp["spatial_connectivities"]`.

**Data** (`gfs/data/hemisphere_new.py`, the current loader) —
- `PyGAnnData` reads pre-split `isocortex_train_final.h5ad` / `isocortex_valtest_final.h5ad`.
- Gene expression and xyz are concatenated into a single `x` tensor; `gene_exp_ind` and
  `xyz_ind` index the two blocks. Models slice `data.x[:, data.gene_exp_ind]`.
- Labels: `subclass`, label-encoded, 22 classes.
- `NeighborLoader` samples 2-hop patches (batch 64); `NeighborLoaderMod` wraps it to tag
  each node with `subgraph_id` = which seed node's patch it belongs to. **This tag is what
  lets the selection mask be sampled per patch rather than per node or per batch.**
- val/test come from the same valtest graph via `val_mask` / `test_mask`.

`hemisphere.py` (693 lines) is the older multi-file / cross-validation loader, kept around
but not used by `antelope.py`.

**Model** (`gfs/models/antelope.py`) —
- `GnnFs.logits` is an `(n_select, gene_ch)` parameter. Training: Gumbel-softmax per row
  (each row picks one gene) per subgraph, then `max` over rows → soft k-hot mask per patch,
  broadcast to nodes by `subgraph_id`. Eval: hard argmax mask. Mask multiplies `x`.
  This is the PERSIST / concrete-selector approach (Jang et al. 2016).
- Backbone: GATv2 / SAGE / GCN stack following "Classic GNNs are strong baselines"
  (Luo et al. 2024) — pre-linear, per-layer residual `lins`, LayerNorm, JK-style summation.
- Two side paths added to the output logits: an MLP on masked expression (`x_res`) and a
  linear map of per-patch mean-centered xyz (`xyz_status`).
- `HalfHop` augmentation (`models/transforms.py`) inserts slow nodes between neighbors;
  their predictions are dropped via `slow_node_mask` before the loss.
- `tau_schedule(type, epoch, total)`: `"old"` = exponential 10 → 0.01. Anything else falls
  through to a two-stage schedule, 10 → 2 over the first 60% of epochs, then → 0.05.

**Trainer** (`gfs/trainers/antelope.py`) — hydra `@main` on `configs/antelope.yaml`,
`seed_everything` + `setup_seeds`, TensorBoard logger, checkpoint on `val_f1_overall` (max),
then `trainer.test(ckpt_path="best")`. Run dirs are `data/logs/<datetime>_<prefix>_s<seed>/`.

Every epoch `on_train_epoch_end` appends selected gene indices and their softmax
probabilities to `selections.csv` in the log dir — this is the primary result artifact.
`on_test_epoch_end` dumps `test_pred.pt` (predictions + labels).

## Model variants (baselines being compared)

Each is a near-copy of `antelope.py` with its own trainer and `scripts/main_*.sh`.
Changing the shared backbone means propagating the edit by hand.

| file | selection method |
|---|---|
| `antelope.py` | concrete / Gumbel k-hot (PERSIST) — the main method. Also has an `fs_method="scGist"` branch: soft gate + `FeatureRegularizer` pushing weights toward 0/1 and their sum toward the panel size |
| `antelope_stg.py` | STG / L0 stochastic gates (`models/stg/layers.py`) |
| `antelope_l1.py` | L1-penalized gating layer — for a fair L1-vs-L0 comparison |
| `antelope_topk.py`, `antelope_topk_nsamples.py` | differentiable top-k via the pluggable samplers in `models/samplers/` |
| `antelope_loc.py` | xyz regression head instead of classification |
| `models/persist.py` | MLP-only PERSIST, no graph — ablation for whether the graph helps |
| `antelope_hardtraining.py` (trainer) | `hard_training=True`: hard mask during training too |
| *(no new file)* `scripts/main_rootloss.sh` | plain `antelope.py` with `model.trainmode=1` — loss on the 64 root nodes only, so the selector sees 74,633 instead of 4.84M `(cell, mask)` pairs/epoch. Tests mask exposure — **confirmed it: the panel drifts off `{12, 83}`.** See "Mask exposure: CONFIRMED" |
| `antelope_shuffle.py` + `data/hemisphere_new_shuffle.py` | negative control — one fixed permutation, see below |
| `antelope_shuffle_epoch.py` + `data/hemisphere_new_shuffle_epoch.py` | same control but the train permutation is redrawn every epoch, see below |
| `antelope_shuffle_within.py` + `data/hemisphere_new_shuffle_within.py` | shuffle control whose **valtest permutation is block-diagonal** (val cells → val nodes, test cells → test nodes). Train is byte-identical to `antelope_shuffle.py`. Used by `scripts/main_shuffle_rootloss.sh`; see "The shuffled mask-exposure arm" below |
| `antelope_panel.py` + `data/panel_tabular.py` | **no selection at all** — scores an already-frozen gene panel with self-edges only, on the upsampled tabular set, at matched gradient parity. This is the in-repo version of the baseline protocol; see "The frozen-panel pipeline" below |
| `antelope_panel_graph.py` + `data/hemisphere_new_panel.py` | frozen panel, **graph kept** (shuffled or intact). Differs from `antelope_shuffle.py` only by `fs_method: "none"`; isolates noise-robustness from annealing. **Both arms run: shuffled 0.6282 (neutral), intact 0.7103 (+0.082)** — see "Panel-graph results" |

## Running

SLURM, from inside `scripts/` (paths in the sbatch files are relative):

```bash
sbatch main.sh   # python -u ../gfs/trainers/antelope.py <hydra overrides>
```

Hydra overrides on the command line are how experiments are configured, e.g.
`data.prefix="select15_seed0_tauold_500ep" model.n_select=15 data.rand_seed=0 model.tautype="old"`.
`data.prefix` names the log dir, so encode the experiment in it.

Cluster note: `main.sh` / `main_shuffle.sh` / `main_shuffle_epoch.sh` / `main_panel.sh` /
`main_panel_graph.sh` / `main_rootloss.sh` / `main_shuffle_rootloss.sh` target Allen
(`-p celltypes`, a100). The other
`main_*.sh` still carry the older GSU/TReNDS header (`-p qTRDGPU`, `-A trends53c17`) and
conda prefix — port those before reusing.

To request a different GPU without editing the sbatch file, override on the command line — CLI
flags beat the in-file `#SBATCH` directives:

```bash
sbatch --gres=gpu:l40s:1 main_panel.sh 0
```

`main_panel.sh` and `main_panel_graph.sh` take positional args instead of raw hydra overrides:
`main_panel.sh [seed] [exposure_dir] [genes] [prefix]` and
`main_panel_graph.sh [seed] [genes] [prefix] [shuffle_cells]`. **Every arg defaults**, so a bare
`sbatch main_panel.sh 0` silently runs the *Persist* panel — pass all of them when you mean a
different one, and confirm from the `panel verified:` / `panel: columns` line in stdout.

There is **no `squeue`/`sacct` on the interactive/dev node** — job state has to be checked from a
login node. Watching for the `data/logs/<datetime>_<prefix>_s<seed>/` dir to appear works as a
proxy for "it started".

## Current state (as of 2026-07-30)

Recent committed work: isocortex train/valtest split, the new two-stage tau schedule, the
L1 baseline, SLURM scripts ported to Allen.

Uncommitted / untracked: the **shuffle negative control** — permutes gene expression and
labels together while leaving xyz and graph connectivity fixed, so each biological cell
lands on a random spatial node. Ran for seeds 0–4 at `n_select=5`, in both the
fixed-permutation and per-epoch-redraw forms; the two are indistinguishable (see below).

Also untracked: the **node-exposure counter + upsampled tabular set** (`gfs/data/node_exposure.py`),
the **frozen-panel tabular pipeline** (`gfs/data/panel_tabular.py`,
`gfs/trainers/antelope_panel.py`, `gfs/configs/antelope_panel.yaml`, `scripts/main_panel.sh`), which
reproduces the external baseline protocol in-repo at matched gradient parity, and the
**panel-graph arm** (`gfs/data/hemisphere_new_panel.py`, `gfs/trainers/antelope_panel_graph.py`,
`gfs/configs/antelope_panel_graph.yaml`, `scripts/main_panel_graph.sh`) — frozen panel with the
graph kept, to test noise-robustness; **both arms now run to completion.** Also
`scripts/main_rootloss.sh` — plain `antelope.py` at
`model.trainmode=1` (no new python files; all hydra CLI overrides), testing mask exposure — and its
shuffled counterpart, the **within-split shuffle mask-exposure arm**
(`gfs/data/hemisphere_new_shuffle_within.py`, `gfs/trainers/antelope_shuffle_within.py`,
`scripts/main_shuffle_rootloss.sh`), which is the better-controlled of the two; **both completed
2026-07-30.** Plus one
functional edit to tracked code: `antelope.py`'s
per-subgraph xyz centring is now vectorized (numerically equivalent; see "Perf fix" below). Note
`antelope.py` also carries a cosmetic `ruff format` sweep on top of pre-existing uncommitted changes.

Per standing instruction, **none of this is committed.**

**The headline, as of 2026-07-30 — there are now TWO separate contributions, not one.** Five overnight
runs (four in this repo + the PERSIST exposure arm) all finished and resolved the two open questions:

1. **The gene-quality gap is real, and mask exposure is how the selector gets it — confirmed in both
   directions.** Persist 0.5748 vs the selector's panel 0.6321 through the *identical* frozen-panel
   pipeline ("The decisive frozen-panel result"). Then: starving GFS of mask draws (`trainmode=1`, job
   24950004) makes its panel **lose `{12, 83}`** → `{2, 38, 225, 435, 480}`, and feeding PERSIST the
   same 4.84M pairs/epoch (job 24948056, in `/home/donghyun.kim/persist/`) makes it **gain them** →
   `[2, 12, 83, 280, 465]`. So the selector's advantage is **sample efficiency**, not the graph and not
   a better selection mechanism: patch expansion is a free 65× multiplier on `(cell, Gumbel mask)`
   pairs. **The PERSIST comparison is therefore mask-budget-confounded** — see "Mask exposure:
   CONFIRMED" and the outstanding 6-minute label-fix control.
2. **The intact graph is worth +0.082 weighted F1 at classification time** — the largest single effect
   measured anywhere in this investigation. Frozen panel, no selector, only the cell permutation
   varying: shuffled 0.6282 → intact **0.7103** (jobs 24943657 / 24943704). Noisy neighbours are
   **neutral** (0.6282 ≈ tabular 0.6321), so this is the graph carrying real signal, not regularization.
   See "Panel-graph results".

**Important framing correction** — the standalone argument, with all evidence in one place, is
`docs/does-the-graph-help.md`. In brief: earlier notes concluded "the graph is irrelevant / GFS's
advantage never came from the graph". That is accurate **only about selection** — the selector does find
`{2, 12, 83, 435, 480}` with the graph destroyed. It is **wrong as a claim about the model overall**:
every arm that looked graph-neutral was comparing *shuffled vs no-graph*, i.e. two ways of having no
usable spatial signal. The first run to compare against an **intact** graph at a fixed panel found a
large benefit. Write it up as: *the selector picks a good panel without spatial context; the graph then
adds an independent +0.08 at classification time.*

**Open loose ends, in priority order:**

1. **The 6-minute label-fix control:** `cd /home/donghyun.kim/persist && sbatch main.sh 5 0 "" 100`
   (raw exposure, fixed labels). The PERSIST exposure run changed *two* things at once — exposure **and**
   a label-encoding fix — so `{12, 83}` isn't yet attributable to exposure alone. **The only outstanding
   run that could still change a conclusion.** Details in the ⚠️ under "Mask exposure: CONFIRMED".
2. **Score PERSIST's matched-exposure panel** `[2, 12, 83, 280, 465]` through the frozen-panel pipeline —
   quantifies the confound in F1 rather than gene identity.
3. `shuffle_cells=false xyz_status=false` — separates "graph structure" from "xyz became usable" in the
   +0.082.
4. An intact `trainmode: 0` `select5` run — the missing cell in the rootloss 2×2.
5. A **random 5-gene panel** as the selection floor, and seeds 1–4 everywhere for error bars.

Result worth keeping in mind: the shuffle control reaches `test_f1_overall ≈ 0.63`,
`test_overall_acc ≈ 0.66`. This was previously described as "essentially unchanged from the
intact-graph runs" — but the only intact comparison available then was `select15` (different panel
size). At **matched** panel size the intact graph scores 0.7103, so the neighborhood contributes a lot;
what stays true is that ~0.63 is achievable from `x_res` alone. Selections converge cleanly
(probs → 0.999); `n_select=15` / 500 epochs stably picks a set including gene indices
2, 5, 12, 38, 83, 435, 480, overlapping the `n_select=5` picks (that run scores 0.8533 weighted).

### Why the shuffle control outscores the tabular baselines (investigated 2026-07-28)

In `f1_side_by_side_with_shuffle_plot.png`, Shuffle at `n_select=5` (weighted F1 ≈ 0.63 /
macro ≈ 0.44) sits *above* Persist (~0.56/0.40), SMASH (~0.56/0.41) and scGIST (~0.50/0.40),
with Selektor at ~0.70/0.52. Shuffle is plotted only at panel size 5; the others span
5/10/15/20. Per-seed shuffle numbers recomputed from `data/logs/*_shuffle_s*/test_pred.pt`
(test n = 31562, 22 subclasses): weighted 0.6279 / 0.6200 / 0.6313, macro 0.4339 / 0.4524 /
0.4386 for seeds 0–2. Prefer recomputing from `test_pred.pt` over reading values off the plot.

**Baseline protocol.** Each external (tabular, no-graph) method selects a panel; that frozen
panel is then scored with a variant of GFS that has the **selection module removed**, the whole
GNN **retrained from scratch** with **self-edges only**. Those runs live in a *separate
directory*, not this repo — which is why no baseline runs appear in `data/logs/`. The protocol
is sound; do not use the `self_loops_only` bug below as an argument against those numbers.

**The shuffle is a control for the graph, not for feature selection.** It is not a
chance-level floor. `hemisphere_new_shuffle.py:126-130` applies one *shared* permutation to
both `gene_exp` and `celltype`, so every cell keeps its own expression↔label pairing and only
the spatial relationship is destroyed.

**The selector is completely unharmed by shuffling — verified.** All 5 shuffle seeds converge
on essentially the same panel, and it is the intact run's panel (from
`data/logs/*_shuffle_s*/selections.csv`, last epoch):

| seed | picked genes | inside intact-15 |
|---|---|---|
| s0 | 2, 12, 83, 435, 480 | 5/5 |
| s1 | 2, 12, 83, 225, 480 | 4/5 |
| s2 | 2, 12, 83, 435, 480 | 5/5 |
| s3 | 2, 12, 25, 83, 480 | 4/5 |
| s4 | 2, 12, 83, 435, 480 | 5/5 |

{2, 12, 83, 480} in all five seeds. **23/25 picks fall inside the intact `select15` panel**, at
probs → 0.999. So shuffling the graph costs the selector *nothing* — it recovers the same
marker genes. This is the empirical version of the shared-permutation point above. Shuffled GFS
is effectively "PERSIST-quality MLP on a well-selected 5-gene panel", so ~0.63 is its expected
score.

**Why does it beat the baselines, then?** Given the protocol is sound, the architecture
asymmetry actually runs *against* shuffle, which sharpens the question. With
`add_self_loops=False` (`antelope.py:165,179`):

- **Baseline model (self-edges only):** the conv's only edge is the self-edge, so attention
  softmaxes over one element = 1.0 and the conv path reduces to a clean linear map of the
  node's own expression. Plus the `x_res` MLP. **Two clean signal paths.**
- **Shuffle model:** conv aggregates random neighbors and has no self-loop, so that path is
  pure noise. Only `x_res` carries signal. **One clean path, plus an active noise injection.**

Shuffle wins despite being handicapped. That leaves two candidates:

1. **The panel is genuinely better.** The concrete selector picks {2, 12, 83, 480} *without any
   help from the graph*. If that beats the tabular baselines' panels under identical readout,
   the honest conclusion is that GFS's selection advantage was never coming from the graph — it
   comes from the selector mechanism (per-patch Gumbel k-hot + joint training with the MLP
   head). Publishable, just not the story the graph framing implies.
   *(Confirmed, with two 2026-07-30 refinements: the selector mechanism's edge is specifically **mask
   sample efficiency** — "Mask exposure: CONFIRMED" — and this claim is about **selection** only. At a
   fixed panel the intact graph is worth +0.082 F1, so don't paraphrase it as "the graph doesn't
   help". See "Panel-graph results".)*
2. **A protocol asymmetry worth ruling out first.** The shuffle F1 is measured **in-run** — the
   same run that learned the panel, weights co-adapted to it over 100 epochs while the mask was
   still moving, and the selection module present at eval (hard argmax). Baselines are
   **retrained from scratch** on a frozen panel. Not the same measurement, and the direction of
   the bias is not obvious a priori.

**The one experiment that separates them.** Take the shuffle panel — e.g.
`[2, 12, 83, 435, 480]` — and run it through the baseline pipeline: no-selection GFS,
self-edges, retrained from scratch. Same treatment the baselines got.

- Still ~0.63 → candidate 1. The selector really finds better genes; the graph is irrelevant.
- Drops toward ~0.56 → candidate 2. The gap was in-run co-adaptation, not gene quality.

*(Ran it: 0.6321, so candidate 1. But "the graph is irrelevant" as phrased here was too broad — it is
true of selection, false of classification at a fixed panel. See "Panel-graph results".)*

Worth adding a **random 5-gene panel** through that same pipeline as the true selection floor,
and the intact `n_select=5` panel too, so every bar is measured identically.

**The baseline directory is not here**, so the protocol above is taken as described — but a
baseline stdout log has since been inspected directly, which answers the config questions and
settles the optimization ones; see "A baseline run, read directly" below.

`halfhop: true` in the shuffle config is a **confirmed no-op under self-edges**:
`models/transforms.py:58-61` isolates self-loops and excludes them from half-hopping, so with
no neighbor pairs the remaining edge set is empty and zero slow nodes are inserted. So HalfHop
is live regularization for shuffle and literally inert for the baselines.

### Per-epoch shuffle variant (set up 2026-07-28, seeds 0–4 completed 2026-07-29)

A second negative control where the training permutation is **redrawn every epoch** instead of
drawn once. Rationale: with a single fixed permutation, each cell keeps one stable (wrong) set
of neighbours for all 100 epochs, so the conv path can in principle memorise that fixed noise.
Reshuffling per epoch makes the neighbourhood signal fresh noise each time, which should
suppress that and leave the `x_res` MLP path as the only usable route.

Three untracked files:

| file | role |
|---|---|
| `gfs/data/hemisphere_new_shuffle_epoch.py` | loader; `reshuffle_train()` at `:177` |
| `gfs/trainers/antelope_shuffle_epoch.py` | trainer; `reload_dataloaders_every_n_epochs=1` at `:82` |
| `scripts/main_shuffle_epoch.sh` | sbatch wrapper; seed from `$1` (default 0), `n_select=5` |

Mechanism:

- `epoch_shuffle_seed(rand_seed, epoch) = rand_seed * 1_000_003 + epoch` (`:19-29`) — a pure
  function of the epoch, **not** a mutable counter. Lightning may call `train_dataloader()`
  more than once for the same epoch; keying on the epoch makes a repeat call a no-op rather
  than a second redraw. The stride decorrelates run-seed 0/epoch 1 from run-seed 1/epoch 0.
- The datamodule keeps **unpermuted reference copies** of `gene_exp` / `celltype` (`:99-100`),
  so each epoch permutes the original ordering rather than compounding permutations.
- `xyz` (and the xyz columns of `x`) and the edgelist are never permuted, same as
  `hemisphere_new_shuffle.py`. Expression and labels still move together under one shared
  permutation, so this is still a graph control, not a feature-selection control.
- **val/test use a single fixed permutation from `rand_seed` and are NOT reshuffled per epoch**
  (`:212-213`), so `val_f1_overall` stays comparable across epochs and checkpoint selection
  remains meaningful.

Reads `configs/antelope.yaml` like every other trainer (no dedicated config), so `halfhop:
true` and `max_epochs: 100` carry over; `n_select` is overridden to 5 in the sbatch script.

**Results (all 5 seeds ran to 100 epochs; `data/logs/*_shuffleepoch_s*/`).** Metrics recomputed
from `test_pred.pt` (test n = 31562, 22 subclasses), panels from the last row of
`selections.csv`.

| seed | F1 weighted | F1 macro | F1 micro (= overall acc) | panel @ ep99 |
|---|---|---|---|---|
| s0 | 0.6296 | 0.4380 | 0.6587 | 2, 12, 83, 435, 480 |
| s1 | 0.6284 | 0.4340 | 0.6578 | 2, 12, 83, 435, 480 |
| s2 | 0.6275 | 0.4293 | 0.6569 | 2, 12, 83, 435, 480 |
| s3 | 0.6170 | 0.4668 | 0.6502 | 2, 12, 83, **207**, 480 |
| s4 | 0.6308 | 0.4327 | 0.6617 | 2, 12, 83, 435, 480 |
| **mean ± sd** | **0.6267 ± 0.0055** | **0.4402 ± 0.0152** | **0.6571 ± 0.0042** | |

All probs ≥ 0.994. `{2, 12, 83, 480}` in all five seeds, `435` in 4/5 (s3 swaps in 207);
**24/25 picks fall inside the intact `select15` panel** {2, 5, 12, 38, 83, 435, 480}.

**Conclusion: the per-epoch redraw changes nothing.** Against the fixed-permutation shuffle
(recomputed the same way: mean weighted 0.6261, mean macro 0.4407, modal panel
`2, 12, 83, 435, 480` in 3/5 seeds), the per-epoch variant matches to within 0.001 weighted —
well inside the ~0.005 seed sd — and its panel is *more* seed-consistent, not less. So the
fixed-shuffle score was **not** the conv path memorising one stable permutation's noise; that
path contributes essentially nothing either way, and ~0.63 comes from the `x_res` MLP on a
well-chosen panel. This closes the open comparison but does not decide candidate 1 vs 2 above —
that still needs the frozen-panel retrain through the baseline pipeline.

Metric-naming trap: `test_overall_acc` is `MulticlassAccuracy(average="weighted")`, which for
accuracy equals macro-recall (0.4398 here), **not** the ~0.66 figure. The number comparable to
"overall accuracy ≈ 0.66" in the fixed-shuffle notes above is F1/accuracy **micro**.

Relaunch command, for reference: `for s in 0 1 2 3 4; do sbatch main_shuffle_epoch.sh $s; done`.

### Why shuffle still beats the baselines, given no spatial context can exist (2026-07-29)

The per-epoch result rules out "the conv path memorised one fixed permutation". Spatial
neighbourhood information provably cannot be reaching the model in either shuffle variant, yet
shuffle scores ~0.63 against Persist/SMASH ~0.56 and scGIST ~0.50 — *with the same GFS
architecture on the baseline side, just self-edges*. Candidates, ranked, with what was verified
in code.

> **ANSWERED — #2 wins.** Read "The decisive frozen-panel result" below before this section. Both
> panels were run through one identical frozen-panel pipeline: Persist 0.5748, selector's panel
> 0.6321. That measurement holds #1, #3 and #4 constant across the two bars, so all three are
> eliminated. The reasoning below is kept for the record and because the mechanisms are still worth
> understanding. (An earlier ranking also led with a training-exposure asymmetry; that is retracted
> below — read it before citing any exposure argument. Note the retraction is scoped to the
> *classifier*: the analogous **mask**-exposure asymmetry on the selector does not cancel — and as of
> 2026-07-30 it is **confirmed**, see "Mask exposure: CONFIRMED".)
>
> **2026-07-30 addendum — the question this section asks is now partly ill-posed.** It assumes the
> baselines are the right reference and asks why shuffle beats them. Two later results reframe it:
> (a) shuffle's edge over Persist is gene quality, and the selector gets that from ~65× more mask
> draws per epoch — so the comparison is mask-budget-confounded, not purely architectural;
> (b) shuffle is a *weak* configuration, not a strong one — the same panel with the graph **intact**
> scores 0.7103 vs shuffle's 0.6282. The premise "no spatial context can exist" is what makes shuffle
> score only 0.63; the interesting comparison was never shuffle-vs-baselines but intact-vs-everything.

**1. Soft-mask annealing is a full-gene warm start.** ELIMINATED. With tau 10 → 0.01 the mask is
soft for most of training, so early epochs pass *all 500 genes* at low amplitude through `res_lin`
(500→128→128→22) and narrow gradually; per-patch Gumbel sampling also makes it a gene-dropout
ensemble. Baselines retrain from scratch with 495 inputs hard-zeroed from step 0. Curriculum
plus implicit regularization — attributable to neither the graph nor gene quality. But the frozen
panel scores 0.6321 *without* any of it, so the curriculum was not the source.

**2. The panel is genuinely better.** **CONFIRMED.** The selector converges to `{2, 12, 83, 480}`
with the graph destroyed in two different ways, so its advantage never needed spatial context — and
its panel scores 0.6321 vs Persist's 0.5748 under an identical readout with no selector present in
either.

**3. Asymmetric regularization.** ELIMINATED (both by the baseline underfitting — nothing for
regularization to fix — and by the frozen-panel run, where HalfHop is off in both arms). HalfHop is
active for shuffle and inert for the baselines (confirmed above). Smaller effect: shuffle's conv path
injects noise into the summed logits, pressuring `x_res` toward robust high-margin solutions, whereas
the baseline's conv path is a clean linear map of own expression that competes with the MLP for the
same signal and can co-adapt on train. The residual-branch noise-injection idea is the one piece of
this still worth a direct test — that is the panel-graph arm, below. **Tested 2026-07-30: dead.**
Frozen panel with shuffled neighbours scores 0.6282 vs 0.6321 with no graph at all — noise injection
into the residual branch is worth nothing, either way.

**4. Gradient-noise scale (weak).** ELIMINATED by direct measurement: the upsampled pipeline puts
both arms at 4147 rows/step and 1167 steps/epoch, and the Persist score moved 0.002. Not exposure
either — see the retraction below — but each shuffle
step averages over ~4,147 nodes (5.6% of the training set) versus 64 (0.086%) at the same
constant lr 1e-3. Because permuted patch membership is independent of cell identity, a shuffle
step's gradient is a near-iid 5.6% sample, much closer to the full-batch gradient, which plausibly
matters for rare subclasses in a 22-class problem. Sign is unclear though: small batches at
constant lr often generalize as well or better, so this is not obviously a baseline handicap.

**RETRACTED: there is no ~65× training-exposure asymmetry.** An earlier version of this section
claimed one, from the fact that `antelope.py:391` sets `idx = data.train_mask` under `trainmode:
0` and `train_mask` is all-ones on the train graph — so every node in the patch contributes a
loss term (~4,147/batch vs 64), i.e. ~4.84M vs 74,633 terms per epoch. That counts loss *terms*,
which get averaged away: `self.loss_ce = nn.CrossEntropyLoss()` (`antelope.py:345`) uses default
**mean** reduction, and both arms take the same 1167 optimizer steps per epoch (74,633/64). Patch
expansion multiplies a cell's appearances *and* the batch-mean denominator by the same factor:

| arm | appearances/epoch | weight in its batch mean | total gradient weight/epoch |
|---|---|---|---|
| shuffle (2-hop, deg 20) | ~64.8 | 1/4,147 | **0.0156** |
| baseline (self-edges) | 1 | 1/64 | **0.0156** |

Exactly equal. What survives is only the gradient-noise difference in #4. Corollary: **do not
"fix" this by running ~65 sweeps per epoch on the baseline** — that would manufacture a real
65× step-count asymmetry where none currently exists.

**Scope of this retraction: it is about the classifier, not the selector.** Mean reduction cancels
gradient *weight*, which is what the table above counts. It does not cancel the number of distinct
`(cell, Gumbel mask)` pairs the selection layer is scored on — patch expansion multiplies mask draws
65× and nothing averages that away, because each repeat carries a *different* mask. See "Mask
exposure" below; that asymmetry is real and still untested.

Patch size and appearance count above are **measured, not estimated** — see "Measured node
exposure" below. An earlier version of this section put them at ~10,000 and ~156 by assuming
independent degree-20 expansion; real 2-hop neighbourhoods overlap heavily, so both were ~2.4×
too high. The cancellation is unaffected (64.8/4,147 = 1/64 exactly).

**The shuffle runs are fully converged, which weakens this whole line further.** From
`data/logs/20260728_193213_select5_seed0_shuffleepoch_s0/lightning_logs/`: `val_f1_overall` over
the last 10 epochs is 0.633/0.632/0.630/0.632/0.627/0.632/0.629/0.630/0.631/0.629 — flat — with
final `train_loss_ce` 1.103 vs `val_loss_ce` 1.140 and `train_overall_acc` 0.654 vs
`val_overall_acc` 0.661. Train ≈ val, no overfitting, on a plateau: the model is at the capacity
ceiling of a 5-gene panel, not cashing in surplus optimization. A converged model does not care
how it got there.

**So the right protocol is to match convergence, not exposure.** Cheaper and more defensible:
read the baseline runs' val curves. If they plateaued with train ≈ val like the above,
optimization is dead as an explanation and the gap is #1 or #2. If they were still descending at
100 epochs they were simply undertrained — retrain with early stopping on `val_f1_overall`. Only
if batch-noise parity is also wanted: raise `data.batch_size` toward ~4,150 (scaling
`max_epochs` to hold step count), or keep 64 and train to early stopping. Safe to change:
`bn: true` is dead config (`antelope.py:280-283` is `if self.ln: … elif self.bn:` and `ln: true`,
so BatchNorm never runs), hence no batch-statistics dependence on batch size.

**Two things that do NOT explain it:**

- *Model capacity on signal favours the baselines.* With `add_self_loops=False`
  (`antelope.py:165,179`) the baseline's single self-edge makes attention softmax over one
  element = 1.0, so its conv path is a clean linear map of own expression — two clean paths,
  versus shuffle's one clean path plus active noise injection. The gap exists despite this.
- *The xyz path isn't it.* `antelope.py:250-252` subtracts a per-patch mean (note: a scalar over
  all 3 coords, `torch.mean` with no `dim`). With 1-node patches the baseline's centred xyz has
  absolute position removed, so the cortical-depth signal that should have favoured the baselines
  is largely destroyed there too.

**What to check — DONE.** The convergence questions were answered by reading a baseline log directly
(next section), and the frozen-panel retrain separating #1 from #2 has now been run in-repo on both
panels: Persist 0.5748, shuffle's `[2, 12, 83, 435, 480]` 0.6321. See "The decisive frozen-panel
result". The in-repo pipeline sidesteps the broken `self_loops_only=True` by building self-edge
batches directly in `panel_tabular.py`'s collator, so that rough edge never needed fixing. Left
undone: a **random 5-gene panel** (the selection floor) and the intact `n_select=5` panel through the
same pipeline, plus seeds 1–4.

### Measured node exposure, and the upsampled tabular dataset (2026-07-29)

`gfs/data/node_exposure.py` counts **exactly** how many loss terms each training cell contributes
per epoch under the graph loader, then turns those counts into an upsampled tabular dataset that
reproduces them. Built so baseline gene panels can be trained/tested on a dataset with the same
per-cell exposure the graph model actually gets, instead of a uniform tabular epoch.

It imports the real `PyGAnnDataGraphDataModule` and iterates the `NeighborLoader` it builds, so
batch partitioning cannot drift from training. It iterates the *inner* loader, skipping
`NeighborLoaderMod`'s `subgraph_id` recomputation — that is expensive and does not change patch
membership. Since `batch.n_id` is unique within a batch, "loss terms per node per epoch" == "number
of batches the node appears in". HalfHop slow nodes are dropped before the loss
(`antelope.py:406`) so they contribute nothing.

**Measured at `batch_size=64`, `n_hops=2`** (the `antelope.yaml` values), mean over 3 epochs:

| quantity | value |
|---|---|
| train cells | 74,633 |
| batches/epoch | 1,167 |
| nodes per batch (patch) | **~4,147** |
| total loss terms/epoch | **4,839,547** |
| appearances per cell/epoch | **64.84** (sd 7.63) |
| min / p1 / median / p99 / max | 31.7 / 46 / 65 / 83.3 / 119.7 |
| never seen | 0 |
| epoch-to-epoch corr | 0.971 |

A *single-seed* 2-hop patch is only ~64 nodes, so 64 seeds sharing one batch overlap heavily —
which is why the ~10,000/~156 figures from assuming independent degree-20 expansion were ~2.4× too
high. Out-degree is exactly 20 by construction but in-degree ranges 1–35 and the graph is **not
symmetric**.

**Exposure is close to uniform across subclasses** (`rel_to_uniform` 0.87–1.05), so the graph does
not meaningfully over-sample abundant classes. Most-underexposed: `029 L6b CTX Glut` 0.875,
`330 VLMC NN` 0.918, `other` 0.920 — sparse/boundary populations. Most-overexposed:
`022 L5 ET CTX Glut` 1.048, `005 L5 IT CTX Glut` 1.039. So the upsampled set is close to, but not
identical to, plain 65× replication.

**Label-encoding trap this handles.** `020 L2/3 IT RSP Glut` is present in train (791 cells) but
**absent from valtest**, so `valtest.obs.subclass` carries only 21 categories against train's 22.
Encoding labels per file silently shifts the integers between splits. The script derives the
encoding from train, writes it to `classes.json`, and applies it to all outputs as an explicit
`label` column with an assertion.

Usage (outputs to `data/exposure/<tag>/`, gitignored):

```bash
python -m gfs.data.node_exposure --epochs 5                      # counts only, ~10s
python -m gfs.data.node_exposure --epochs 5 --genes 2,12,83,435,480 \
    --materialize --emit-valtest --tag panel5_shuffle             # full artifacts
```

`node_exposure_counts.csv` (per-cell, per-epoch), `node_exposure_summary.json`, `classes.json`,
`upsampled_index.npy` (int32 row index into train cells), and with the flags
`upsampled_train.h5ad` + `tabular_val.h5ad` / `tabular_test.h5ad`. `--divisor N` scales the dataset
down while preserving *relative* exposure; `repeats = round(count/divisor)` with stochastic
remainder rounding so expected repeats equals the true count. At `divisor=1.0`: 4,839,281 rows
(64.84×), ~500 MB for 5 genes, ~10 GB for all 500.

**val/test are deliberately NOT upsampled** — `validation_step`/`test_step` score root nodes only
(`antelope.py:472,565`), so each val/test cell is already seen exactly once per epoch; upsampling
them would only reweight the metric.

Caveat when using this: the upsampled train set delivers ~65× the gradient steps per epoch, and the
baseline already plateaus by epoch 5 of 100 (next section). Scale `max_epochs` down rather than
running 100 epochs over it.

### A baseline run, read directly (2026-07-29)

`out-comms-3521173.out` (repo root, untracked) is the stdout of a real baseline evaluation —
**the Persist seed-0 panel scored with self-edges-only GFS**. It resolves everything that was
previously "not checkable from this repo".

Config from its header dump:

```
data.genes: [2, 250, 435, 480, 481]   n_genes: 5   self_loops_only: true
batch_size: 512   max_epochs: 100   lr: 0.001 constant   xyz_status: false
halfhop: true   trainmode: 0   prefix: persist_select5seed0
data_root = /data/users1/dkim195/graphFeatureSelect/data/     # the separate GSU dir
```

So: `self_loops_only: true` **runs fine there** — the crash in Known rough edges is specific to
this repo's copy of the loader. Also note `xyz_status: false` for the baseline vs `true` for
shuffle, and `batch_size: 512` vs 64 (146 train batches/epoch).

**Test metrics:** weighted F1 **0.5724**, macro F1 **0.4013**, micro 0.6079, macro_acc 0.3955,
loss 1.285 — matching the ~0.56/0.40 Persist bar in the plot, so the plotted baselines are real
and reproducible.

**It is thoroughly converged, and converges almost immediately** (per-epoch metrics parsed from
the progress-bar postfix; 100 epochs present, 0–99):

| epoch | train_loss | val_loss | train_acc | val_acc | val_f1 |
|---|---|---|---|---|---|
| 0 | 1.720 | 1.350 | 0.511 | 0.591 | 0.538 |
| 5 | 1.350 | 1.310 | 0.588 | 0.599 | 0.558 |
| 50 | 1.320 | 1.300 | 0.595 | 0.599 | 0.559 |
| 99 | 1.310 | 1.300 | 0.598 | 0.601 | 0.565 |

Best `val_f1_overall` = 0.566 at epoch 87; **epoch 5 is already within 0.01 of it**. Epochs 5→99
buy 0.007 F1.

**What this kills:**

- **Undertraining is dead as an explanation.** 94 epochs of visible plateau. More epochs, more
  sweeps, larger batches, early stopping — none of it moves this number. This is the direct
  evidence for the retraction above: both arms plateau, so how they got there does not matter.
- **The baseline underfits, it does not overfit.** `train_loss ≥ val_loss` for the *entire* run
  (1.310 vs 1.300 at the end) and train_acc ≈ val_acc ≈ 0.60. The model has saturated what those
  5 genes contain — a panel-capacity ceiling, not a generalization failure. So regularization
  asymmetry (#3, HalfHop) had nothing to fix, and the remaining explanation has to be about the
  genes.

**Which makes #2 (the panel is genuinely better) the leading candidate.** Both arms converge,
both plateau, both underfit, and they land at 0.572 vs 0.627 weighted. The panels differ by
exactly two genes:

| | panel |
|---|---|
| Persist | 2, 250, 435, 480, **481** |
| shuffle | 2, **12**, **83**, 435, 480 |

Shared {2, 435, 480}; Persist adds {250, 481} where the selector picks {12, 83} — and {12, 83}
appear in 5/5 shuffle seeds *and* inside the intact `select15` panel. Honest framing: GFS's
selector finds a better 5-gene panel than Persist, and does so **with no spatial information at
all**.

**Residual caveat, and the experiment that closed it — RESOLVED.** This is the frozen-panel protocol
applied to *Persist's* panel, while shuffle's 0.627 was still measured in-run. That caveat is gone:
`[2, 12, 83, 435, 480]` has since been pushed through an in-repo equivalent of this pipeline and
scored **0.6321** — slightly *above* its in-run 0.627, so co-adaptation was not inflating anything.
See "The decisive frozen-panel result" two sections down. Still open: a random 5-gene panel as the
selection floor.

### The frozen-panel pipeline, in-repo (built 2026-07-29)

Three untracked files reproduce the baseline protocol here, so panels can be scored without the
separate GSU directory:

| file | role |
|---|---|
| `gfs/data/panel_tabular.py` | serves the upsampled train set + plain val/test as **self-edge** PyG batches |
| `gfs/trainers/antelope_panel.py` | hydra entrypoint; **imports** `LitGnnFs` from `antelope.py` so the backbone cannot drift |
| `gfs/configs/antelope_panel.yaml` | `fs_method: "none"`, `batch_size: 4147`, `xyz_status: false`, `halfhop: false` |
| `scripts/main_panel.sh` | sbatch wrapper: `main_panel.sh [seed] [exposure_dir] [genes] [prefix]` |

`fs_method: "none"` is a new 3-line branch in `GnnFs.__init__` that creates **no `logits`
parameter** (asserted in the trainer). `forward` already no-ops the mask for unrecognised
`fs_method`, since both selection branches are `if`, not `else`.

**Gradient parity is exact, by construction.** `batch_size=4147` on the 4.84M-row upsampled set
gives 1167 optimizer steps/epoch — the same as the graph run, with the same 1/4147 per-row weight
in the mean-reduced loss:

| | graph arm | panel arm |
|---|---|---|
| optimizer steps/epoch | 1167 | 1167 |
| rows per step | ~4147 (measured patch mean) | 4147 |
| per-row weight in batch mean | 1/4147 | 1/4147 |
| gradient weight/cell/epoch | 0.015637 | 0.015637 |
| per-cell appearances/epoch | 64.84 (measured) | baked into the upsampled rows |

`ln: true` (and `bn` dead) means normalisation is per-row, so no cross-row statistics couple
samples — random-row batches vs spatially-correlated patches changes gradient *noise*, not the
expected gradient. One residual asymmetry: with `shuffle=True` a cell's ~65 duplicate rows scatter
independently, so it lands in ~63.2 distinct batches rather than exactly 65 (~1.8 collisions/epoch).
Total weight per cell per epoch is unchanged; only its per-step lumpiness is, on the order of the
graph run's own patch-size variation (561–4425). Note the graph loader also uses `shuffle=True`
(`hemisphere_new.py:241`), so that part is matched.

**What deliberately differs** — these *are* the definition of the baseline, not drift:

- no selection module → **no soft-mask annealing warm start, no per-patch Gumbel gene-dropout
  ensembling**. That is candidate #1, the thing being tested.
- `gene_ch=5` not 500, so `res_lin` is `MLP(5→128→128→22)` and `lin_in` is 5→32. **35,884 params
  vs the graph model's 120,172.**
- `xyz_status: false` → `xyz_lin` does not exist (matches `out-comms-3521173.out`). The most
  arguable of these, though the graph arm's scalar centring makes xyz near-useless anyway.
- `halfhop: false` — genuinely equivalent, just explicit rather than silently inert.
- `subgraph_id = arange(n)` (per-cell patches) vs ~4147-node patches.

**Result: the training regime was never the confound.** 100 epochs of the Persist panel
`[2, 250, 435, 480, 481]` through this pipeline (job 24940498, 30.4 min on an L40S; recomputed from
`test_pred.pt`, n = 31562):

| | weighted F1 | macro | micro |
|---|---|---|---|
| this pipeline, upsampled + matched regime (100 ep) | 0.5748 | 0.4040 | 0.6097 |
| original baseline (`out-comms-3521173.out`, batch 512, 100 ep) | 0.5724 | 0.4013 | 0.6079 |
| shuffle GNN target | **0.6267** | 0.4402 | 0.6571 |

Upsampling + regime-matching moved it **0.002**, not the 0.055 needed to reach shuffle. It
reproduces the *old baseline number*. Consistent with that run plateauing by epoch 5 and
underfitting throughout — there was no optimization deficit for a better regime to recover. So
**do not frame this as "matching the regime closes the gap"; the data contradicts that.** The
result is the negative one: candidates #1/#3/#4 are eliminated by direct measurement.
(An earlier note here quoted 0.5707 from a 3-epoch partial read of the same run; the finished
100-epoch value is 0.5748.)

### The decisive frozen-panel result: the panel is genuinely better (2026-07-29)

The same pipeline on **shuffle's own panel** `[2, 12, 83, 435, 480]` (job 24941387, dataset
`data/exposure/panel5_shuffle/`, 30.5 min). The two runs differ in **nothing but which two genes
fill slots 2–3** — same code, same upsampled-exposure regime, same 100 epochs, same seed, same
`batch_size=4147`/1167 steps, selector absent in both.

| panel | weighted F1 | macro | micro | best `val_f1_overall` |
|---|---|---|---|---|
| Persist `[2, 250, 435, 480, 481]` | 0.5748 | 0.4040 | 0.6097 | 0.5669 @ ep1 |
| **shuffle `[2, 12, 83, 435, 480]`** | **0.6321** | **0.4352** | **0.6641** | 0.6290 @ ep9 |
| (shuffle GNN, in-run, 5-seed mean) | 0.6267 | 0.4402 | 0.6571 | — |

**This confirms candidate #2 and closes the investigation.** Two conclusions:

1. **The gap is gene quality.** 0.5748 → 0.6321 from swapping `{250, 481}` for `{12, 83}` under an
   identical readout. Nothing about the graph, the annealing curriculum, HalfHop, batch noise, or
   exposure survives as an explanation *for the classifier* — every one of those is now held constant
   across the two bars. (This does not close the separate question of what let the *selector* find
   `{12, 83}` in the first place — **now largely answered: mask exposure.** See "Mask exposure:
   CONFIRMED" below.)
2. **Co-adaptation contributed nothing.** Shuffle's frozen-panel score (0.6321) is *slightly above*
   its own in-run score (0.6267), not below. Had the in-run number been inflated by weights
   co-adapting to a still-moving soft mask, retraining from scratch on the frozen panel would have
   dropped toward 0.57. It didn't.

Honest framing for a writeup: **GFS's selector finds a better 5-gene panel than Persist, and does so
with no spatial information at all** — the selection mechanism (per-patch Gumbel k-hot + joint
training with the MLP head) is the contribution, not the graph.

**Updated 2026-07-30, two amendments — read these before quoting the sentence above:**

- *"...not the graph"* is right about **selection** and wrong about the **model**. At this same frozen
  panel, un-permuting the cells takes 0.6282 → 0.7103 (+0.082). The graph is a large independent
  contributor at classification time; it just isn't what finds the genes. See "Panel-graph results".
- *"the selection mechanism is the contribution"* now has a specific mechanism behind it: **mask sample
  efficiency**, not the k-hot parameterization per se. Starving the selector of mask draws costs it
  `{12, 83}`. See "Mask exposure: CONFIRMED".

So the two-contribution framing: *the selector finds a better panel because patch expansion gives it
~65× more mask samples per epoch for free; the graph then adds +0.08 F1 on top at classification time.*

Gene identities (`isocortex_train_final.h5ad` `var_names` are Ensembl IDs; symbols resolved
separately): col 2 = **Grik3**, 12 = **Ptprm**, 83 = **A830036E02Rik**, 435 = **Slc30a3**,
480 = **Dscaml1**; Persist's extra picks 250/481 = ENSMUSG00000023959 / ENSMUSG00000019828. Shared
`{2, 435, 480}` = Grik3/Slc30a3/Dscaml1, so the entire 0.057 F1 gap reduces to **Ptprm +
A830036E02Rik vs those two**. Reviewer-bait to pre-empt: `A830036E02Rik` is an uncharacterized RIKEN
cDNA/lncRNA, and it is in 5/5 shuffle seeds.

**Both panel runs mildly overfit late**, unlike the reference baseline:

| | Persist ep0 → ep99 | shuffle ep0 → ep99 |
|---|---|---|
| train_loss | 1.382 → 1.240 | 1.212 → 1.055 |
| val_loss | 1.301 → **1.380** | 1.120 → **1.207** |
| val_f1 | 0.5612 → 0.5608 (peak 0.5669 @ep1) | 0.6209 → 0.6199 (peak 0.6290 @ep9) |

val_loss turns upward within a few epochs while train_loss keeps falling — a consequence of 1167
steps/epoch on the upsampled set vs the reference's 146 at `batch_size=512`. **Harmless here**:
`ModelCheckpoint` monitors `val_f1_overall` (max) and `trainer.test(ckpt_path="best")`, so both test
numbers come from the peak-val epoch (1 and 9), not epoch 99. But if this pipeline is reused, either
add early stopping or cut `max_epochs` to ~20 — 95% of the run is wasted, and quoting a final-epoch
number would understate both arms by ~0.006.

**Still missing for a publication figure**: a **random 5-gene panel** through this same pipeline (the
selection floor — without it a reviewer cannot tell 0.57 vs 0.63 from noise around "any 5 genes get
~0.6"), and seeds 1–4 on both panels for error bars. The seed sd on the shuffle GNN was ~0.005,
which is 10× smaller than the 0.057 gap, so the conclusion is not fragile — but the bars need it.

Startup guard: `check_panel()` asserts the `exposure_dir`'s `var_names` match `data.genes`, so
pointing at a dir built for a different panel fails immediately. It does **not** catch a fully
defaulted invocation (`sbatch main_panel.sh 0` silently runs the *Persist* panel), so check the
`panel verified:` line in stdout. `ENSMUSG00000033278` / `ENSMUSG00000084890` in positions 2–3
means shuffle; `...023959` / `...029151` means Persist.

### Mask exposure: CONFIRMED as a real mechanism (arms completed 2026-07-30)

> **ANSWERED — mask exposure is real, confirmed in both directions.** Both rootloss arms ran to 100
> epochs overnight: **the panel loses `{12, 83}` when the selector is starved of mask draws.** And the
> mirror-image arm (PERSIST fed the same 4.84M pairs/epoch, job 24948056) **gains** them. Results in
> "Rootloss results" immediately below; the setup reasoning follows it. Net effect: the gene-quality
> conclusion stands (it was measured under a frozen panel), but *how* GFS finds those genes is now
> answered — ~65× cheaper access to `(cell, mask)` pairs — which makes the PERSIST comparison
> mask-budget-confounded. One label-fix control still outstanding; see the Current-state summary.

Every experiment above measures the *classifier* under a fixed panel. None of them measures how the
**selector** got its panel — and there the 65× patch expansion does **not** cancel.

#### Rootloss results (both arms, 100 epochs, 2026-07-30)

Panels from the last row of `selections.csv`; metrics recomputed from `test_pred.pt` (n = 31562).

| arm | job | panel @ ep99 | probs | weighted F1 | macro | micro |
|---|---|---|---|---|---|---|
| shuffle, `trainmode 0` (5-seed control) | — | **2, 12, 83, 435, 480** | ≥0.999 | 0.6267 ± 0.0055 | 0.4402 | 0.6571 |
| **shuffled rootloss**, `trainmode 1` | 24950004 | **2, 38, 225, 435, 480** | 0.969–0.999 | **0.6037** | 0.4240 | 0.6319 |
| intact rootloss, `trainmode 1` | 24944818 | **5, 83, 207, 225, 480** | 0.975–0.999 | 0.6100 | 0.4565 | 0.6337 |

**The one-variable comparison is row 1 vs row 2** (shuffled cells in both; only `trainmode` differs).
Reading it:

- **The panel drifts.** Starving the selector swaps `{12, 83}` → `{38, 225}`. Gene 12 (`Ptprm`) — one
  of the exact two genes the "decisive frozen-panel result" attributes the whole 0.057 F1 gap to — is
  **lost**, and `225` is a gene no `trainmode: 0` run ever converged on. Only `{2, 435, 480}` survive,
  which is precisely the set Persist *also* found. So under mask starvation GFS's panel degrades
  toward the baseline's panel.
- **Probs still converge** (0.969–0.999), so this is not a failure-to-anneal artifact — the selector
  confidently commits to a *different, worse* panel.
- **F1 drops 0.023** (0.6267 → 0.6037), directionally consistent with the panel being worse, though
  in-run F1 conflates panel quality with the 65×-smaller gradient signal to the classifier. The panel
  is the cleaner readout.

**Verdict: mask exposure was (at least partly) the mechanism.** Per the pre-registered reading above —
"panel drifts → mask exposure **was** the mechanism, and the fair comparison against the tabular
baselines is this run" — the honest framing becomes: *GFS's selector finds better genes than PERSIST
in large part because patch expansion hands it ~65× more `(cell, Gumbel mask)` samples per epoch at no
extra optimizer cost.* That is still a genuine advantage of the graph-patch formulation, but it is a
**sample-efficiency** story, not a "the selector mechanism is better" story, and it is *not* about
spatial information.

**This does not touch the frozen-panel conclusion.** 0.5748 vs 0.6321 compared two already-chosen
panels under an identical selector-free readout; `{12, 83}` really is the better pair. Mask exposure
explains *how GFS found them*, which is exactly the question that section left open.

**The PERSIST exposure arm also finished, and it confirms the mechanism from the other side.** Job
24948056 (`/home/donghyun.kim/persist/`, completed 2026-07-30 01:04, 100 epochs / 4.3 h): feeding
PERSIST the same 4.84M `(cell, mask)` pairs/epoch makes it pick **`[2, 12, 83, 280, 465]`** — **it finds
both `12` and `83`**, the exact two genes the 0.057 F1 gap was attributed to, versus `[2, 250, 435, 480,
481]` at raw exposure.

| direction | manipulation | panel | `{12, 83}`? |
|---|---|---|---|
| starve GFS | `trainmode=1`, 74,633 pairs | `2, 38, 225, 435, 480` | **lost** |
| GFS as-is | 4.84M pairs | `2, 12, 83, 435, 480` | both |
| **feed PERSIST** | `--exposure`, 4.84M pairs | **`2, 12, 83, 280, 465`** | **gained** |
| PERSIST as-is | 74,633 pairs | `2, 250, 435, 480, 481` | neither |

**Both directions agree, so the mechanism is confirmed: the selector's gene-quality edge is mask sample
efficiency.** GFS's patch expansion is a free 65× multiplier on Gumbel mask draws, and selection quality
is bottlenecked on those draws — not on the graph, and not on the k-hot parameterization being a better
selector than PERSIST's.

**This changes the writeup materially:** the published-PERSIST-vs-GFS panel comparison is
**mask-draw-budget-confounded**. At matched budget PERSIST recovers `{2, 12, 83}`. Frame the
contribution as *"the graph-patch formulation multiplies mask samples for free, and that is what buys
panel quality"* — do not present 0.5748 as PERSIST's ceiling.

⚠️ **One control still outstanding before citing this.** The PERSIST exposure run carries *both* the
exposure change **and** a label-encoding bug fix (valtest `.cat.codes` per-file, which corrupted the val
loss driving LR scheduling / best-model restore); the historical bar has neither. So `{12, 83}` is not
yet attributable to exposure alone. The separating run is 6 minutes:
`cd /home/donghyun.kim/persist && sbatch main.sh 5 0 "" 100` (raw exposure, fixed labels). Also
unvalidated: `[2, 12, 83, 280, 465]` has not been scored through the frozen-panel pipeline, and this is
one seed. See `/home/donghyun.kim/persist/CLAUDE.md` → "RESULT (job 24948056)".

Caveat, and it is a real one: **the intact arm's panel `{5, 83, 207, 225, 480}` keeps gene 83 and
scores slightly *higher* (0.6100)** than the shuffled arm. Since there is still no intact
`trainmode: 0` `select5` run to compare it against (see the caveats below), that row cannot be read as
a `trainmode` effect on its own. It does weakly suggest the intact graph partly compensates for lost
mask draws. Running intact `select5` at `trainmode: 0` would close this, and is now the cheapest
loose end in the investigation.

Both arms show a **severe early-training pathology** worth knowing about before reusing `trainmode=1`:
`val_loss_ce` explodes to 30–50 while `train_loss_ce` collapses to ~0.17 by epoch 10, then both
recover and `val_f1` climbs from ~0.15 to ~0.60 only after epoch ~50. With 64 loss terms/batch the
model badly overfits the root nodes while tau is still high. It does recover, and `val_f1` was still
creeping up at epoch 99 (best 0.6107 @ep86 intact, 0.6004 @ep99 shuffled) — **so unlike every other
arm here, these two are not fully converged, and 100 epochs may understate them.** Note the
`trainmode: 0` shuffle run shows the same spike shape (val_loss 64 → 100 → recovery), so it is a
property of the annealing schedule, not of `trainmode`.

The retraction above is about gradient *weight*, and it holds: `CrossEntropyLoss` is mean-reduced, so
each cell's total per-epoch weight is 0.0156 under both `trainmode: 0` (whole ~4147-node patch) and a
64-term baseline. What it does not address is **how many distinct `(cell, Gumbel mask)` pairs the
selection layer is scored on**:

| | loss terms/epoch | distinct (cell, mask) pairs seen by `logits` |
|---|---|---|
| GFS, `trainmode: 0` | 4,839,547 | ~4.84M |
| tabular baseline / `trainmode: 1` | 74,633 | n/a — no stochastic layer |

`get_mask` (`antelope.py:219-238`) draws a fresh per-subgraph Gumbel k-hot every forward pass, so a
cell reappearing in another patch arrives **under a different mask**. For a deterministic classifier
that repeat is pure redundancy — exactly why upsampling the tabular baseline moved it only 0.002
("Regime matching does NOT close the gap"). For a stochastic selector each repeat is a genuinely new
sample of the combinatorial mask space. Mean reduction cancels the weight, not the *information*.

So this is compatible with the frozen-panel result rather than a competitor to it. That measurement
compared two *already-chosen* panels under an identical readout and stands: `{12, 83}` beats
`{250, 481}`. Mask exposure is a candidate explanation for **how the selector found the better
panel** — cheap access to millions of mask draws, not anything about the graph. Same headline
("the selector is the contribution"), but a concrete mechanism, and a falsifiable one.

**The test: `scripts/main_rootloss.sh`** — `antelope.py` with `model.trainmode=1`, so the loss covers
only the 64 root nodes (`antelope.py:403-404`, a pre-existing branch no run in `data/logs/` had ever
used). `n_select=5`, `tautype="old"`, 100 epochs, seed 0, intact graph. Nothing in `gfs/` was
modified; it is all hydra CLI overrides. **Completed 2026-07-30 04:22** (job 24944818, 100 epochs,
~2:50/epoch, `data/logs/20260729_215101_select5_seed0_rootloss_s0/`) — results above.

**Measured, not assumed:** `trainmode=1` really does index exactly 64 rows. Root count per batch over
3 real batches: 64/64/64 (all unique) against patch sizes 3997/4211/4236. So 74,633 loss terms/epoch,
one per training cell.

**Only the loss is restricted — the forward pass is not.** The whole ~4,100-node patch still goes
through the model, so each root's prediction is built from its real 2-hop neighbourhood; only the
gradient signal narrows. And **evaluation was already root-only in both arms** —
`validation_step`/`test_step` hardcode `idx = where(n_id == input_id)` (`antelope.py:482,573`) with no
`trainmode` branch — so `val_f1_overall` and the test metrics stay directly comparable to the
`trainmode: 0` runs. `train_loss_ce` does **not** (see the `batch_size` caveat below).

**Read `selections.csv`, not just F1.** The panel is the readout. This was the pre-registered reading,
and **the second branch is what happened** (see "Rootloss results" above):

- panel still lands inside `{2, 12, 83, 435, 480}` at probs → 0.999 → mask exposure was **not** what
  found those genes; the selector is sample-efficient and the gene-quality conclusion needs no
  qualification.
- **panel drifts**, or probs stall below ~0.99 → mask exposure **was** the mechanism, and the fair
  comparison against the tabular baselines is this run, not the `trainmode: 0` one. ← **this one.**
  The panel drifted (`{12, 83}` → `{38, 225}`) while probs still converged.

Caveats for this arm:

- **No 65× speedup.** `trainmode` only changes which rows are indexed for the loss; the forward pass
  still covers the whole patch and `NeighborLoaderMod`'s `subgraph_id` loop (96% of step time) is
  untouched. **Measured: ~2:50/epoch, 4.7 h/100 epochs** — the ~5–7 h estimate was about right.
- **`batch_size` is logged as `train_mask.sum()`** (`antelope.py:395`) regardless of trainmode, so it
  reports ~4147 while 64 terms enter the loss. Cosmetic, but it makes epoch-averaged `train_loss_ce`
  non-comparable across trainmodes.
- **The matching control does not exist for the *intact* arm.** There is no intact-graph
  `trainmode: 0` `select5` run in `data/logs/` — the completed `select5` runs are all shuffle. Nearest
  reference is `select15_seed0_tauold_500ep`, whose panel {2, 5, 12, 38, 83, 435, 480} is a superset of
  the shuffle picks. Either run intact `select5` at `trainmode: 0` too, or use the shuffled arm
  (`main_shuffle_rootloss.sh`), which already has its 5-seed control.
- **Don't read a verdict off the first ~10 epochs.** Every `trainmode: 0` `select5`/`select15` run
  also sits on `{393, 38, 472, 280, 228}` through ~epoch 10 and only migrates to
  `{2, 12, 83, 435, 480}` later in annealing. Verified across `*_shuffle_s0` and both
  `select15_seed0_tauold` runs.
- **`tautype` is the config trap.** `tau_schedule` falls through to the two-stage schedule for *any*
  value that is not the literal string `"old"`, and `main_shuffle.sh` never passed it — so the 5
  completed shuffle runs inherited whatever `antelope.yaml` held at the time. **Checked 2026-07-29:**
  `data/logs/20260721_180328_select5_seed0_shuffle_s0/lightning_logs/version_0/hparams.yaml` shows
  `tautype: old`, so pinning `"old"` in both rootloss scripts keeps the comparison one-variable rather
  than silently changing the schedule.
- Solution symmetry that does **not** hold: "train the *frozen-panel classifier* on the upsampled
  dataset" was already run — that *is* the 0.5748 result. It cannot be the fix, for the reason above.

  **But the symmetric experiment on the *selector* side is a different question, and is now built.**
  `/home/donghyun.kim/persist/main.py --exposure bs64_hop2_seed42` feeds the same
  `upsampled_index.npy` to **PERSIST's selection layer**, whose `BinaryMask` draws a fresh Gumbel
  k-hot **per row** (`persist/layers.py:83`), so 4.84M rows/epoch == 4.84M distinct `(cell, mask)`
  pairs/epoch — exactly matching GFS `trainmode: 0`. The 0.002 null result does **not** carry over:
  there the repeated rows fed a *deterministic* classifier (pure redundancy), here they feed a
  *stochastic* selector (genuinely new mask samples). 100 exposure-epochs = 483.9M mask draws ≈ GFS's
  484.0M over 100 epochs. **RAN 2026-07-30 (job 24948056): panel `[2, 12, 83, 280, 465]` — it found
  both `12` and `83`.** Per the pre-registered reading — "if the exposure arm's panel moves toward
  `{12, 83}`, mask exposure **was** the mechanism, and the honest framing becomes *PERSIST underperforms
  because it sees 65× fewer mask samples per epoch, not because the selector is worse*" — **that is
  exactly what happened.** This is the mirror image of `main_rootloss.sh` (which starves GFS and loses
  `{12, 83}`), so both directions agree. See the Current-state summary above and
  `/home/donghyun.kim/persist/CLAUDE.md` → "RESULT (job 24948056)" for the outstanding label-fix
  confound control.

  Independent of the above: the PERSIST panels in `persist_genes_selected_genes.csv` were produced
  with a **label-encoding bug** (valtest `.cat.codes` computed per-file, mislabeling 60% of val cells,
  which corrupted the val loss PERSIST uses for LR scheduling and best-model restore). Fixed
  2026-07-29 in `persist/main.py`, but the csv — and therefore the 0.5748 bar — predates the fix.

### The shuffled mask-exposure arm, and the within-split permutation (completed 2026-07-30)

`scripts/main_shuffle_rootloss.sh` is `main_rootloss.sh` with the cells shuffled: `trainmode=1`,
`n_select=5`, `tautype="old"`, 100 epochs, seed from `$1` (default 0). **This is the
better-controlled of the two rootloss arms** — the 2×2 it completes, now with results:

| | intact graph | shuffled cells |
|---|---|---|
| `trainmode 0` (4.84M mask pairs) | *no `select5` run exists* | `main_shuffle.sh` — 5 seeds, **0.6267**, `{2,12,83,435,480}` |
| `trainmode 1` (74,633) | 0.6100, `{5,83,207,225,480}` (24944818) | **0.6037, `{2,38,225,435,480}`** (24950004) |

The shuffle row is complete, so that pairing is one-variable; the intact row still is not (no intact
`trainmode: 0` `select5` run exists — the cheapest remaining loose end). The shuffle arm is also where
the question actually lives — those 5 seeds produced the panel that scored 0.6321 in the frozen-panel
run. **Its verdict: the panel drifts off `{12, 83}` under mask starvation, confirming mask exposure as
a real mechanism** — see "Rootloss results" above for the full reading. Completed 2026-07-30 05:45,
100 epochs at ~2:48/epoch, `data/logs/20260729_231747_select5_seed0_shufflerootloss_s0/`.

**Why a new loader was needed.** It runs `antelope_shuffle_within.py`, not `antelope_shuffle.py`. The
original permutes valtest cells across the whole 63,124-cell file. That is **not a leak** — `perm` is
a bijection and val/test *partition* the file, so val nodes get 31,562 distinct cells and test nodes
get exactly the complement, 0 overlap (verified). But it **re-partitions which cells constitute each
split**, swapping ~half of each side (measured: 15,742/31,562 of shuffled-val are true val cells at
seed 0). `hemisphere_new_shuffle_within.py` permutes *within* val and *within* test separately, so both
splits hold exactly the cells the h5ad assigns them while the spatial arrangement inside each is still
destroyed.

| | `hemisphere_new_shuffle.py` | `hemisphere_new_shuffle_within.py` |
|---|---|---|
| train permutation | full, all 74,633 train cells | **inherited, byte-identical** |
| valtest permutation | full, across the val/test boundary | **block-diagonal** (val→val, test→test, plus a defensive "neither" block) |
| `val_mask` / `test_mask` | never permuted | never permuted |
| `xyz`, `edge_index` | never permuted | never permuted |
| dataloaders | `num_workers=8`, generator seed 42 | inherited verbatim → identical patch composition |

Train needs no special handling: `train_mask` is all-ones on the train graph, so the parent's full
permutation is already "within train", and `get_pygdata_obj_train` is inherited unchanged.
`_within_split_perm` asserts the bijection and both block boundaries at runtime.

**Verified against the parent** (`rand_seed=0`, and the perm checks at seeds 0/1/42):

- train `x`, `celltype`, `edge_index` — `torch.equal` **True** (byte-identical)
- valtest `xyz`, `edge_index`, `val_mask`, `test_mask` — `torch.equal` **True**
- valtest expression differs from the parent (expected), and the multiset of val-node expression
  matches the true val cells exactly, same for test, while ~100% of nodes hold a *different* cell
- smoke test (1 epoch, 3 train / 2 val batches, `trainmode=1`) exits 0, writes `selections.csv`,
  `test_pred.pt` `(31562, 22)` + `(31562,)`, and a checkpoint. Metrics are garbage by design
  (`test_f1_overall` 0.006 at tau=10) — it is a plumbing check.

Nothing edits `hemisphere_new_shuffle.py`, so the 5 completed shuffle runs stay untouched.

**One non-strictness to state in a writeup:** those 5 seeds used the cross-split permutation, so their
F1 is not *exactly* comparable. Expected to be tiny — val and test are statistically interchangeable
here (subclass distributions agree to 4e-4 L1, xyz means to 3 decimals) — and the **panel**, which is
the actual readout, is set by training and unaffected either way.

### The panel-graph arm: frozen panel with the graph KEPT (both arms completed 2026-07-30)

> **RESULT — the intact graph is worth +0.082 weighted F1, the largest single effect in this whole
> investigation.** Both arms ran to 100 epochs. `shuffle_cells=true` → 0.6282 (noisy conv is
> **neutral**, as predicted); `shuffle_cells=false` → **0.7103** (intact graph helps, a lot). Full
> numbers in "Panel-graph results" below. **This partially reverses the framing "the graph is
> irrelevant"** — see the reconciliation there, because it does *not* contradict the earlier results.

The tabular pipeline above removes the selector **and** the graph at once. That is fine for the
gene-quality question, but it cannot test a separate hypothesis: **does training against random
neighbours regularize the node representations?** With `add_self_loops=False` and `res: true`
(`antelope.py:289`, each conv has a parallel `Linear` bypass), the shuffle model is structurally
noise injection into a residual branch — a plausible robustness mechanism, and one both existing
arms confound with annealing.

Four untracked files, differing from `antelope_shuffle.py` in **exactly one way**: `fs_method:
"none"`.

| file | role |
|---|---|
| `gfs/data/hemisphere_new_panel.py` | **subclasses** `hemisphere_new_shuffle.py`'s loader; slices `gene_exp` to the panel, `shuffle_cells` switch, `num_workers` respected (parent hardcodes 8) |
| `gfs/trainers/antelope_panel_graph.py` | imports `LitGnnFs`; `report_panel()` prints gene symbols; asserts `not hasattr(model.model, "logits")` |
| `gfs/configs/antelope_panel_graph.yaml` | `genes: [2,12,83,435,480]`, `shuffle_cells: true`, `batch_size: 64`, `n_hops: 2`, `xyz_status: true`, `halfhop: true`, 100 ep |
| `scripts/main_panel_graph.sh` | `main_panel_graph.sh [seed] [genes] [prefix] [shuffle_cells]` |

Note the config matches the **shuffle arm** (`xyz_status: true`, `halfhop: true`, batch 64), *not*
the tabular baseline (`false`/`false`/512). That is deliberate — the whole point is a one-variable
difference from the shuffle GNN. Exposure needs no upsampling here: it uses the same
`NeighborLoader`, so per-cell exposure is the graph's own ~64.84 loss terms/epoch by construction.

The arm table, and how to read the result against shuffle GNN 0.627 / tabular panel 0.632:

| run | annealing | noisy conv | measures | result |
|---|---|---|---|---|
| shuffle GNN (`antelope_shuffle.py`) | yes | yes | — | 0.6267 |
| **this, `shuffle_cells=true`** | **no** | **yes** | **noise-robustness alone** | **0.6282 — neutral** |
| this, `shuffle_cells=false` | no | intact graph | "does the graph help at all, given a frozen panel" | **0.7103 — yes, +0.082** |
| `panel_tabular.py` (self-edges) | no | no | — | 0.6321 |

The pre-registered reading was: **~0.63 → the noisy conv path is neutral**; **>0.64 → noisy neighbours
genuinely add robustness**; **<0.61 → the noise actively hurts**. The shuffled arm landed at 0.6282,
squarely in the **neutral** bucket. And the `shuffle_cells=false` control — flagged at the time as "the
more interesting of the two, since it is the only run in the whole investigation that asks whether the
intact graph helps at all under a frozen panel" — delivered the largest effect in the investigation.
See "Panel-graph results" below.

**Verified byte-identical to the shuffle loader** apart from the panel slice: train/valtest
`edge_index` identical (1,567,293 train edges), `celltype` identical (same permutation), `xyz`
identical, all masks identical, `gene_exp == shuffle's X[:, genes]`, 500 → 5 genes, patch sizes mean
4142 (graph arm 4147), 36.0 K params.

Runtime ~5.4 h/100 ep, of which **96% is `NeighborLoaderMod`'s `subgraph_id` python loop** (161.8 of
166 ms/step). Deliberately not optimized: it is shared with 5 completed shuffle runs and touching it
would put those numbers in question. Prefer `sbatch --gres=gpu:l40s:1 main_panel_graph.sh ...`.
Actual: **2:41/epoch shuffled (4.5 h), 4:57/epoch intact (8.2 h)**. Cause not established — both arms
were A100-PCIE-40GB on different nodes (n10 vs n02) with byte-identical `edge_index`, so HalfHop inserts
the same slow nodes in both and cannot explain it. Most likely node-level contention, not a property of
the arm.

#### Panel-graph results (both arms, 100 epochs, 2026-07-30)

Panel `[2, 12, 83, 435, 480]` in both, seed 0, `fs_method: none` (no selector, verified by the
trainer's assert). Metrics recomputed from `test_pred.pt` (n = 31562).

| run | selector | graph | weighted F1 | macro | micro | best val_f1 |
|---|---|---|---|---|---|---|
| shuffle GNN, 5-seed mean | anneals | shuffled | 0.6267 ± 0.0055 | 0.4402 | 0.6571 | — |
| `panel_tabular.py`, self-edges | none | none | 0.6321 | 0.4352 | 0.6641 | 0.6290 @ep9 |
| **panel-graph, `shuffle_cells=true`** (24943657) | none | shuffled | **0.6282** | 0.4358 | 0.6558 | 0.6370 @ep13 |
| **panel-graph, `shuffle_cells=false`** (24943704) | none | **intact** | **0.7103** | **0.5282** | **0.7281** | 0.7083 @ep41 |

**Two findings, and they are cleanly separated:**

1. **Noisy neighbours are neutral.** 0.6282 vs the tabular 0.6321 and the shuffle GNN's 0.6267 — all
   three within ~0.005, i.e. inside the seed sd. Hypothesis #3's residual noise-injection-regularizes
   idea is **dead**: training against random neighbours neither helps nor hurts. It lands in the
   pre-registered "~0.63 → neutral" bucket.
2. **The intact graph is worth +0.082 weighted / +0.093 macro.** 0.6282 → 0.7103 from *nothing but
   un-permuting the cells*. This is the only run in the investigation that asked "does the intact graph
   help at all under a frozen panel", and the answer is emphatically yes — a bigger effect than the
   0.057 gene-quality gap.

**Reconciling this with "the graph is irrelevant" — it is not a contradiction.** The two claims live in
different places and both hold:

- **Selection does not need the graph.** The selector converges on `{2, 12, 83, 435, 480}` with the
  graph destroyed two different ways (fixed and per-epoch shuffle), 23/25 and 24/25 picks inside the
  intact `select15` panel. That is unchanged.
- **Classification does benefit from the graph, substantially.** Every earlier arm that looked
  graph-neutral compared *shuffled vs no-graph* — two ways of having no usable spatial signal. None of
  them compared against an **intact** graph at a fixed panel. This run did, and found +0.082.

So the correct writeup framing is **two separate contributions**, not one: *the selector finds a good
panel without needing spatial context, and the graph then adds a large independent boost at
classification time.* The earlier phrasing "GFS's selection advantage was never coming from the graph"
is still accurate as written — it is about **selection** — but do **not** paraphrase it as "the graph
doesn't help", which this result falsifies.

Caveat to state honestly: the intact arm keeps `xyz_status: true`, so some of the +0.082 could be the
xyz path becoming informative once patches are spatially coherent (with shuffled cells, a patch's cells
are scattered, so per-patch centred xyz is noise; intact, it encodes local geometry). The scalar-mean
centring quirk limits how much xyz can carry, but this is not fully disentangled. A
`shuffle_cells=false xyz_status=false` run would separate "graph structure" from "xyz became usable" —
worth doing before this number goes in a figure. Also note the reference bar for comparison: intact
`select15` (all 500 genes available, 15 selected) scores 0.8533, so 0.7103 on 5 genes with the graph
sits sensibly between the 5-gene no-graph 0.63 and the 15-gene ceiling.

Both arms mildly overfit late (shuffled: val_loss 1.113 → 1.192 while train falls 1.262 → 1.066;
intact: 0.981 → 0.958 with val_f1 peaking at ep41), and `ckpt_path="best"` on `val_f1_overall` recovers
the peak in both cases, so the test numbers are not final-epoch. Same ~20-epoch advice as the tabular
pipeline applies if reused.

### Perf fix: vectorized per-subgraph xyz centring (2026-07-29)

`antelope.py:250-262` — the per-subgraph xyz mean-centring was a python loop over
`subgraph_id.unique()`, now an `index_add_` scatter-mean. It **preserves the existing quirk** of
using a single scalar mean over all 3 coords (the original called `torch.mean` with no `dim`), so
graph-arm numbers do not shift.

Cost of the loop scaled with subgraph *count*: fine at 64 patches, pathological when every node is
its own subgraph (the panel arm's 4147).

| | old | new |
|---|---|---|
| panel arm, 4147 patches | 673 ms | **0.15 ms** |
| graph arm, 64 patches | 10 ms | 0.15 ms |
| graph full forward (`xyz_status=True`, 4876 nodes) | 13.15 ms | 2.12 ms |
| panel 100-epoch projection | 40.8 h | **0.68 h** (24.5 s/epoch measured) |

Equivalence verified against a subclass restoring the original loop verbatim, weights copied:

- forward, graph config, eval and train/Gumbel paths: **2.1e-07** relative
- gradients: **6.7e-07** max relative over all 27 tensors; `logits` grad 4.4e-07
- 60 training steps with tau annealing, identical seeds: max **5.7e-07** relative loss divergence,
  identical final eval acc, **identical selected genes**, max prob diff 7.3e-07
- float64 exact to 1e-14 → the float32 residual is summation-order noise
- also checked ragged patches, gaps in subgraph ids, single-patch

The new version no longer mutates its `xyz` argument in place. That is safe: the trainer passes
`xyz=data.x[:, data.xyz_ind]`, and advanced (tensor) indexing returns a **copy, not a view**, so
the old in-place write only ever hit a per-step temporary and never reached `data.x`.

Left alone deliberately: `torch.where(n_id == input_id.unsqueeze(-1))` (lines ~396/473/537/564)
builds a 4147² bool matrix in the panel arm vs 64×4147 in the graph arm, but that is 0.12 ms on an
L40S — not worth touching shared code. And `subgraph_id = arange(n)` stays in `PanelCollator`: now
that it is free, it is the semantically correct choice and stays correct if `xyz_status: true` is
ever wanted there. The dataloader was never the bottleneck (1.8 ms/batch at `num_workers=8`).

## Known rough edges

- `FeatureRegularizer` non-strict branch calls `torch.max(tensor, 0)` where
  `torch.clamp(min=0)` is meant; `torch.max` with a dim returns a namedtuple, so
  `strict=False` would break.
- `self_loops_only=True` crashes in this repo's loader. `hemisphere_new.py:54-68` assigns
  `self.adj` in that branch, but the consumers read `self.adj_train` (line 101) and
  `self.adj_valtest` (line 135), which are only ever assigned in the `else` branch →
  `AttributeError` in `setup()`. Same in `hemisphere_new_shuffle.py:58`. Second defect: the
  identity matrix is sized `adata_train.shape[0]²` with no valtest counterpart, so fixing the
  attribute name alone still leaves val/test mis-sized. All seven `gfs/configs/*.yaml` ship
  `self_loops_only: false`, and no run in `data/logs/` used it. (The baseline self-edge
  evaluation lives in a separate directory and is unaffected — `out-comms-3521173.out` shows
  `self_loops_only: true` running to completion there, so this is purely a local defect.)
- `hemisphere_new.py` hardcodes `num_workers=8` and dataloader generator seed 42, ignoring
  `config.data.num_workers` and `config.data.rand_seed`. Same in `hemisphere_new_shuffle.py:308-309`
  and therefore in `hemisphere_new_shuffle_within.py`, which inherits the loaders on purpose so patch
  composition matches the completed shuffle runs. Consequence: patch composition is identical across
  seeds; only the permutation and the weight init vary with `rand_seed`.
- **Shuffling valtest across the val/test boundary re-partitions the splits.**
  `hemisphere_new_shuffle.py` / `_shuffle_epoch.py` permute all 63,124 valtest cells at once. That is
  not a leak (bijection over a set val/test partition → 0 overlap, verified), but ~half of each split
  swaps sides, so "val" and "test" are not the cell sets the h5ad defines. Use
  `hemisphere_new_shuffle_within.py` for new work; the 5 completed shuffle seeds and the 5
  shuffle-epoch seeds predate it.
- Not bugs, but easily-missed behaviours when comparing runs across graph densities: with
  `trainmode: 0` the loss covers **every node in the sampled patch** (`antelope.py:391`,
  `train_mask` is all-ones on the train graph), so patch size — and hence effective batch size and
  gradient noise — scales with graph degree. Measured: ~4,147 nodes/batch and 64.84 appearances per
  cell per epoch at `batch_size=64`, `n_hops=2` (`gfs/data/node_exposure.py`). It does **not**
  change total gradient weight per cell per epoch, because `CrossEntropyLoss` is mean-reduced (see
  the retraction in the section above). It **does** change how many `(cell, Gumbel mask)` pairs the
  selection layer is scored on — 65× more, and mean reduction does not cancel that because each
  repeat carries a different mask. `trainmode: 1` is the switch that removes it, and no run in
  `data/logs/` had used that branch before `scripts/main_rootloss.sh` /
  `scripts/main_shuffle_rootloss.sh`. `trainmode` affects **training only**: it restricts which rows
  enter the loss (measured: exactly 64 unique roots/batch against patch sizes ~4000–4240), not the
  forward pass, which still covers the whole patch. `validation_step`/`test_step` hardcode
  `idx = where(n_id == input_id)` (`antelope.py:482,573`) with no `trainmode` branch, so **eval was
  already root-only in every arm** — which is also why `panel_tabular.py` deliberately does not
  upsample val/test.
  `HalfHop` inserts nothing when self-loops are the only edges (`models/transforms.py:58-61`), so
  `halfhop: true` silently means "off" in that regime.
- `bn: true` is dead config: `antelope.py:280-283` is `if self.ln: … elif self.bn:` and every
  config ships `ln: true`, so the `bns` ModuleList is built but never applied.
- `forward`'s per-patch xyz centring subtracts **one scalar across all three coordinates** rather
  than a per-axis centroid (the original `torch.mean(xyz[...])` had no `dim`). Probably not
  intended. The loop is now vectorized (`antelope.py:250-262`) but the quirk was **deliberately
  preserved**, so fixing it is still an open behaviour change that would shift every result.
- `GnnFs.reset_parameters` calls `self.logits.reset_parameters()`, which doesn't exist on a
  `Parameter`; also unconditionally touches `res_lin` / `xyz_lin`, which may not exist. Now also
  broken for `fs_method: "none"`, where `logits` isn't created at all. Nothing calls it.
- Several configs (`antelope_topk.yaml`, `antelope_cerebellum.yaml`, the region-specific
  ones) still reference the older `file_names` / `test_names` / `cv` data schema that
  `hemisphere_new.py` doesn't read.
- Debug `print`s remain scattered through model/loader code, some commented out.
- `LitGnnFs.on_test_epoch_end` reads `self.logger._root_dir` (`antelope.py:647`), so
  `Trainer(logger=False)` raises `AttributeError: 'NoneType'` at the end of `.test()`. Pass a real
  logger even for throwaway smoke tests.
- Anything O(n_subgraphs) is a latent trap for the panel arm, where `subgraph_id = arange(n)` makes
  n_subgraphs equal the batch size (4147) rather than 64. The xyz centring loop was one such;
  profile before assuming a new per-subgraph loop is cheap.
- `max_epochs: 100` is wrong for the upsampled panel pipeline: at 1167 steps/epoch it peaks on
  `val_f1_overall` by epoch 1–9 and then overfits for 90+ epochs (val_loss climbs while train_loss
  falls). Harmless because `ModelCheckpoint`/`ckpt_path="best"` recover the peak epoch, but a
  final-epoch number would understate the result by ~0.006 and 95% of the runtime is wasted. Cut to
  ~20 epochs or add early stopping if reusing it.
- `NeighborLoaderMod`'s `subgraph_id` recomputation is a python loop over input nodes and costs
  **161.8 of 166 ms/step (96%)** in every graph-arm run — the single biggest speedup available
  anywhere in the repo. Left alone on purpose: it is shared with the 5 completed shuffle runs, and
  changing it would put those numbers in question.

## Conventions

- ruff, line length 120, isort with `gfs` as first-party; pre-commit is configured.
- The naming is `antelope` throughout for the GNN+selection model family — no significance
  beyond being the project codename.
