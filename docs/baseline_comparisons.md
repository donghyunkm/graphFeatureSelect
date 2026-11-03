# Baseline comparison: PERSIST panel with NO graph vs GFS with the full graph

**Status:** scaffolded 2026-08-31; self-edges arm datasets built, not yet launched. Results tables are
placeholders, to be filled from `test_pred.pt` (never quote F1 off a plot — memory
`recompute-metrics-from-test-pred`).

## Question
Give the regenerated PERSIST select-5 panels **test metrics** (PERSIST itself computes none — its driver
only selects genes), and ask: how much of the GFS full-graph F1 is the **panel** vs the **graph**? We
score each PERSIST panel with a **no-graph (self-edges only) classifier** — the same `LitGnnFs` backbone,
no selection module — and compare to the GFS **full-graph 4-hop** model on identical data.

This is the graph-vs-no-graph contrast *at a frozen, externally-chosen panel*. It complements
`docs/feature_ablation.md` (frozen PERSIST panel **with** the 4-hop graph, learned-selector vs frozen).

## Design
Full data (500 genes → sliced to the 5-gene panel, 22 subclasses). Five PERSIST panels (seeds 0–4),
scored by the self-edges baseline; compared to the GFS full-graph **4-hop** arm.

| Arm | trainer | config | graph | selection | panel | train exposure |
|---|---|---|---|---|---|---|
| **PERSIST + no graph** | `antelope_panel.py` | `antelope_panel.yaml` | self-edges only | `fs_method=none` | frozen PERSIST select-5 (per seed) | **1× (non-upsampled)** |
| **PERSIST + shuffled graph** | `antelope_panel_graph.py` | `antelope_panel_graph.yaml` | 2-hop NeighborLoader over a **shuffled** graph (`shuffle_cells=true`) | `fs_method=none` | frozen PERSIST select-5 (per seed) | graph (~65×/cell) |
| **GFS + full graph** (reference) | `antelope_4hop.py` | `antelope_4hop.yaml` | 4-hop NeighborLoader (intact), `trainmode=2` complete-field loss | `fs_method=persist`, `n_select=5` | learned end-to-end | graph (~65×/cell) |

The GFS full-graph reference is the **4-hop** arm from `docs/4hop_2hop_comparison.md` (job `25507593`).
**✅ This arm is now complete (2026-09-01):** all 5 seeds finished 100 epochs, recomputed F1 =
logged `test_f1_overall`. Reference = **0.7099 ± 0.0142 weighted / 0.5272 ± 0.0141 macro** (seeds 0–4).
(The 2026-08-31 "divergence" was a transient mid-tau-anneal snapshot, not a bug — see `4hop_2hop_comparison.md`.)
**Loss-node set matches the frozen-panel arms:**
although sampling is 4-hop, `trainmode=2` takes the loss only on nodes within 2 hops of a seed — the same
within-2-hop set the 2-hop shuffle arm trains on (message passing is 2 layers in both). The extra hops 3–4
carry no loss and only complete those nodes' receptive fields, so per-cell loss exposure is unchanged; the
4-hop reference is comparable to the 2-hop frozen-panel arms.

### Self-edges arm details
- **No graph:** `edge_index = [[0..n-1],[0..n-1]]` (self-loops), `add_self_loops=False` → attention
  softmax over one element (=1.0), so the conv path is a clean linear map of each cell's own 5-gene
  expression. Same backbone, training loop, and eval code as the graph arm (`gfs/data/panel_tabular.py`).
- **Non-upsampled (1×):** each train cell appears exactly once (74,633 rows), built with
  `node_exposure.py --no-upsample`. This is the *vanilla* tabular regime, **not** exposure-matched to the
  graph run's ~65×/cell. Prior work shows exposure/upsampling moves PERSIST F1 only ~0.002 (memory
  `regime-matching-does-not-close-baseline-gap`), so 1× is a fair, simpler baseline. **Caveat:** the arms
  therefore differ in both graph *and* train-exposure regime; the intended read is "panel-only, no graph".
- **`rand_seed=0` for all 5 runs:** each run already differs only by its PERSIST panel, so fixing the
  model seed removes init/shuffle noise as a confound (the panel is the only thing that varies). The GFS
  reference arm varies seed 0–4 because there the seed is what produces its five distinct learned panels
  — so compare the arms **as distributions over their five runs, not paired seed-for-seed.**
- `batch_size=512` on the 1× train (config default 4147 targets the ~4.8M-row upsampled set; on 74,633
  rows it would be ~18 steps/epoch), giving ~146 optimizer steps/epoch. `xyz_status=false`, `halfhop=false`,
  `trainmode=0`, 100 epochs, lr=1e-3 constant, best-val-F1 checkpoint for test.

### Shuffled-graph arm details
- **Shuffled graph = ~65× exposure without real neighbours.** Same `LitGnnFs` backbone, frozen PERSIST
  panel (`fs_method=none`), 2-hop `NeighborLoader`, `batch_size=64` — but the cells are **permuted**
  (`shuffle_cells=true`, `gfs/data/hemisphere_new_panel.py`): `gene_exp` and `celltype` are permuted
  *together* (each cell keeps its own expression↔label pairing, so no leak), while the edges/xyz are left
  in place. A cell's graph neighbours are therefore **random other cells** — no real neighbour signal. But
  the NeighborLoader patches still overlap, so each cell contributes the graph's own ~64.84 loss
  terms/epoch (memory `measured-node-exposure-and-upsampling`). This is what the user's framing captures:
  the shuffled-graph arm is essentially **the panel at graph-level exposure (~65×), minus the real
  neighbour signal** — a noisy-conv upsampling arm, sitting between the self-edges 1× floor and the intact
  graph. (Because `res=true` gives each conv a parallel `Linear` bypass, the noisy conv path is structural
  noise injection into a residual branch, not pure garbage — the model can partly route around it.)
- **`halfhop=false`, `xyz_status=true`** to match the GFS full-graph reference (job 25507593, `antelope_4hop.yaml`,
  HalfHop off) and avoid the eval-time HalfHop perturbation (memory `eval-time-halfhop-active`). **Caveat:**
  this differs from the older `panel_graph` shuffle runs (memory `panel-graph-arm-noise-robustness`, which
  used the config default `halfhop=true`) — do not cross-compare those F1 numbers.
- **`rand_seed=0` for all 5 runs.** `rand_seed` also seeds the cell permutation, so the shuffled graph is
  **identical across the 5 runs** — a fixed graph ablation, panel the only thing varying (fixed vs per-epoch
  shuffle is equivalent, memory `per-epoch-shuffle-matches-fixed-shuffle`).
- **Companion control (not run here):** `shuffle_cells=false` on the same panels is the intact-graph
  frozen-panel arm — the difference (intact − shuffled) is the value of the *real* neighbour signal at a
  frozen panel. Memory `panel-graph-arm-noise-robustness` already estimates this at the selector's panel
  (shuffled 0.6282 vs intact 0.7103, +0.082) but with `halfhop=true`; re-run with `halfhop=false` if a
  matched intact control is wanted.

### The PERSIST panels (select-5, 0-indexed; regenerated post-label-fix, jobs 25507771-5)
| seed | panel (gene-column indices) | gene symbols | exposure_dir |
|---|---|---|---|
| 0 | 2, 12, 174, 435, 480 | Grik3, Ptprm, Tmem215, Slc30a3, Dscaml1 | `exposure/persist_1x_seed0` |
| 1 | 2, 25, 83, 207, 225 | Grik3, Slc17a6, A830036E02Rik, Zfp536, Nxph1 | `exposure/persist_1x_seed1` |
| 2 | 2, 83, 250, 375, 480 | Grik3, A830036E02Rik, Clic5, Kcnmb2, Dscaml1 | `exposure/persist_1x_seed2` |
| 3 | 2, 12, 25, 115, 435 | Grik3, Ptprm, Slc17a6, Sorcs3, Slc30a3 | `exposure/persist_1x_seed3` |
| 4 | 2, 83, 86, 271, 435 | Grik3, A830036E02Rik, Sulf2, Tgfbr2, Slc30a3 | `exposure/persist_1x_seed4` |

## How to run
```bash
# Datasets (already built 2026-08-31): one non-upsampled dir per panel
python -m gfs.data.node_exposure --genes <panel> --no-upsample --materialize --emit-valtest \
    --tag persist_1x_seed<i>

# Self-edges arm (array 0-4, task id -> panel index, rand_seed=0 for all):
cd /home/donghyun.kim/graphFeatureSelect/scripts
sbatch main_panel_persist_1x.sh

# Shuffled-graph arm (array 0-4, task id -> panel index, rand_seed=0, halfhop off):
sbatch main_panel_graph_persist_shuffle.sh
```
Log dirs: `data/logs/<datetime>_persist_1x_panel{0-4}_s0/` (self-edges);
`data/logs/<datetime>_persist_shufgraph_panel{0-4}_s0/` (shuffled graph).

## Runs / Slurm IDs
| Arm | panel | Slurm job | log dir | status |
|---|---|---|---|---|
| PERSIST + no graph | 0 | | | dataset built, not launched |
| PERSIST + no graph | 1 | | | dataset built, not launched |
| PERSIST + no graph | 2 | | | dataset built, not launched |
| PERSIST + no graph | 3 | | | dataset built, not launched |
| PERSIST + no graph | 4 | | | dataset built, not launched |
| PERSIST + shuffled graph | 0 | | | not launched |
| PERSIST + shuffled graph | 1 | | | not launched |
| PERSIST + shuffled graph | 2 | | | not launched |
| PERSIST + shuffled graph | 3 | | | not launched |
| PERSIST + shuffled graph | 4 | | | not launched |
| GFS + full graph (4-hop) | seed 0–4 | 25507593_{0-4} | see 4hop_2hop doc | ✅ done — 0.7099 ± 0.0142 w / 0.5272 ± 0.0141 m |

## Results
_(recompute weighted + macro F1 from `test_pred.pt`, n = 31,562 test cells; never quote off a plot)_

GFS full-graph columns are the **4-hop** reference (job 25507593) — **now complete** (2026-09-01), filled
below. The two PERSIST arms remain to be launched.

| panel/seed | no-graph wtd | no-graph macro | shuf-graph wtd | shuf-graph macro | GFS 4-hop wtd | GFS 4-hop macro |
|---|---|---|---|---|---|---|
| 0 | | | | | 0.6849 | 0.5056 |
| 1 | | | | | 0.7185 | 0.5318 |
| 2 | | | | | 0.7145 | 0.5246 |
| 3 | | | | | 0.7190 | 0.5297 |
| 4 | | | | | 0.7125 | 0.5442 |

Summary (mean ± sd over the five runs):

| Arm | weighted (mean ± sd) | macro (mean ± sd) |
|---|---|---|
| PERSIST + no graph (self-edges, 1×) | | |
| PERSIST + shuffled graph (~65×, noisy conv) | | |
| GFS + full graph (4-hop, intact) | 0.7099 ± 0.0142 | 0.5272 ± 0.0141 |

## Findings
_(fill in after both arms complete. The three arms decompose the GFS full-graph F1 at a frozen PERSIST
panel into stacked contributions:_
- _**no-graph (self-edges, 1×)** = panel-only floor (no exposure, no neighbours);_
- _**shuffled graph (~65×, noisy conv)** − no-graph = the effect of graph-level **exposure + noisy-conv
  regularization** with NO real neighbour signal (the "essentially upsampling" arm);_
- _**GFS full graph (4-hop, intact)** − shuffled graph = the value of the **real neighbour signal**. The
  4-hop reference is comparable to the 2-hop frozen-panel arms: `trainmode=2` takes loss on the same
  within-2-hop node set (2-layer message passing in both), so per-cell loss exposure matches. The 4-hop
  reference is complete: 0.7099 ± 0.0142 w / 0.5272 ± 0.0141 m — the two PERSIST arms still need launching._

_Compare as distributions over the five runs, not paired seed-for-seed (seed handling differs, see Design).
Relate to memory `panel-graph-arm-noise-robustness` (intact − shuffled ≈ +0.082 at the selector panel, but
`halfhop=true` there — not directly comparable), `regime-matching-does-not-close-baseline-gap` (exposure
alone moves PERSIST ~0.002), and `frozen-panel-confirms-panel-quality`.)_
