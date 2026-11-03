# Does graph neighbor information help?

**Short answer: yes for classification (+0.082 weighted F1), no for gene selection.** These are two
separate questions and they have opposite answers. Conflating them produced a months-long wrong
conclusion in this project, so the split is the whole point of this document.

All numbers are weighted F1 on the isocortex test split (n = 31,562, 22 subclasses), recomputed from
`test_pred.pt` rather than read off any plot. Log dirs and SLURM job IDs are given so every row is
traceable.

---

## The claim, stated precisely

| question | answer | strength |
|---|---|---|
| Does the graph help the **classifier** predict cell type? | **Yes, +0.082 weighted F1** | one seed, one direct A/B |
| Does the graph help the **selector** choose genes? | **No — measurably nothing** | 10 seeds, 2 independent ablations |

Both are supported below. The second was established first, which is why it dominated earlier writeups;
the first was not measured until 2026-07-30.

---

## Evidence 1 — the graph helps classification (+0.082)

The decisive experiment is a single-variable A/B. Frozen 5-gene panel `[2, 12, 83, 435, 480]`, **no
selector at all** (`fs_method: "none"`, asserted at startup), identical code, identical config, seed 0,
100 epochs. The **only** difference is whether the cell→node assignment is permuted.

| run | graph | weighted F1 | macro | micro | job | log dir |
|---|---|---|---|---|---|---|
| `panelgraph_shuffle` | shuffled cells | 0.6282 | 0.4358 | 0.6558 | 24943657 | `20260729_214544_panelgraph_shuffle_s0` |
| `panelgraph_intact` | **intact** | **0.7103** | **0.5282** | **0.7281** | 24943704 | `20260729_214945_panelgraph_intact_s0` |

**Δ = +0.082 weighted, +0.092 macro.** This is the largest single effect measured anywhere in the
investigation — larger than the 0.057 gene-quality gap that was previously the headline.

Why this A/B is trustworthy:

- **The loaders are byte-identical apart from the permutation.** Verified with `torch.equal`:
  `edge_index` (1,567,293 train edges), `xyz`, `val_mask`, `test_mask`, and the label vector all match
  between arms. Only the cell→node assignment differs.
- **No selector in either arm**, so annealing, Gumbel gene-dropout and mask co-adaptation are absent
  from both. `hemisphere_new_panel.py` differs from `hemisphere_new_shuffle.py` by `fs_method: "none"`
  plus the panel slice, nothing else.
- **Exposure is matched by construction** — same `NeighborLoader`, so ~64.84 loss terms per cell per
  epoch on both sides (measured, `gfs/data/node_exposure.py`).
- **Both used the same GPU model** (A100-PCIE-40GB, different nodes).
- Test metrics come from the peak-`val_f1_overall` checkpoint (`ckpt_path="best"`), ep13 and ep41, not
  from a final epoch where both mildly overfit.

### The one caveat that is not yet closed

Both arms run `xyz_status: true`, so the model has a side path taking per-patch mean-centred xyz
(`antelope.py:249-262`). With shuffled cells a patch's members are scattered across the section, so that
centred xyz is noise; intact, it encodes local geometry. **Some unknown fraction of the +0.082 is
therefore "xyz became usable" rather than "message passing became usable."**

Two reasons this doesn't sink the claim, and one run that would settle it:

- The centring subtracts a **single scalar mean over all three coordinates** (the original
  `torch.mean` had no `dim`), not a per-axis centroid — so the xyz path is crippled by construction and
  can carry only limited information.
- Both mechanisms are *graph neighbor information* in the sense that matters here: both require cells to
  sit next to their true spatial neighbors. The headline "spatial context helps classification" holds
  either way; only the attribution between the two paths is open.
- **The settling run:** `sbatch main_panel_graph.sh 0 "[2,12,83,435,480]" panelgraph_intact_noxyz false`
  with `model.xyz_status=false`. Do this before the number goes in a figure.

### Context: how +0.082 compares

| configuration | genes | graph | weighted F1 |
|---|---|---|---|
| frozen panel, self-edges only (tabular) | 5 | none | 0.6321 |
| frozen panel, shuffled graph | 5 | noise | 0.6282 |
| frozen panel, intact graph | 5 | **yes** | **0.7103** |
| full selector run, `select15`, 500 ep | 15 | yes | 0.8533 |

0.7103 on five genes sits sensibly between the five-gene no-graph score and the fifteen-gene ceiling.

---

## Evidence 2 — the graph does *not* help gene selection

Three independent lines, all pointing the same way.

**(a) Destroying the graph costs the selector nothing.** Five seeds of the shuffle control at
`n_select=5` converge on the intact run's panel:

| seed | panel | inside intact `select15` |
|---|---|---|
| s0 | 2, 12, 83, 435, 480 | 5/5 |
| s1 | 2, 12, 83, 225, 480 | 4/5 |
| s2 | 2, 12, 83, 435, 480 | 5/5 |
| s3 | 2, 12, 25, 83, 480 | 4/5 |
| s4 | 2, 12, 83, 435, 480 | 5/5 |

`{2, 12, 83, 480}` in **5/5 seeds**; 23/25 picks fall inside the intact `select15` panel
`{2, 5, 12, 38, 83, 435, 480}`, all at probs → 0.999.

**(b) Redrawing the permutation every epoch changes nothing.** Fresh noise each epoch (so the conv path
cannot memorize one fixed wrong neighborhood) gives mean weighted 0.6267 ± 0.0055 across 5 seeds versus
0.6261 for the fixed permutation, and a *more* seed-consistent panel: 24/25 picks inside `select15`.

**(c) The selector's real advantage is mask sample efficiency, not the graph.** Confirmed in both
directions:

| manipulation | `(cell, mask)` pairs/epoch | panel | `{12, 83}`? |
|---|---|---|---|
| GFS starved — shuffle arm (`trainmode=1`, job 24950004) | 74,633 | `2, 38, 225, 435, 480` | **lost** |
| GFS as-is | 4.84M | `2, 12, 83, 435, 480` | both |
| **PERSIST fed** (`--exposure`, job 24948056) | 4.84M | `2, 12, 83, 280, 465` | **gained** |
| PERSIST as-is | 74,633 | `2, 250, 435, 480, 481` | neither |

`{12, 83}` (Ptprm, A830036E02Rik) are the exact two genes the 0.5748 → 0.6321 frozen-panel gap reduces
to. Starving GFS loses them; feeding PERSIST the same budget makes it find them. What the graph
contributes to *selection* is not spatial information — it is that patch expansion makes a cell reappear
in ~65 patches per epoch, each under a fresh Gumbel mask (`antelope.py:219-238`), at no extra optimizer
cost. Mean-reduced `CrossEntropyLoss` cancels the gradient *weight* of those repeats but not the mask
*samples*.

> Caveat on (c): the PERSIST exposure run changed exposure **and** a label-encoding fix simultaneously.
> The 6-minute separating control (`cd /home/donghyun.kim/persist && sbatch main.sh 5 0 "" 100`) has not
> been run. If the label fix alone moves the panel, (c) needs rewriting — (a) and (b) are unaffected.

### Root-loss (starved) runs — both graph arms, with test metrics

Restricting the loss to root/seed nodes only (`model.trainmode=1`, ~74,633 loss terms/epoch vs ~4.84M)
drifts the panel off `{2, 12, 83, 435, 480}` in **both** graph conditions, and both starved runs still
converge confidently (all selection probs ≈ 0.97–0.999 at the final epoch). Test metrics below are read
from each run's `.out` (the Lightning `Test metric` table); `test_f1_overall` = weighted F1. Single seed (s0).

| run | graph | (cell,mask)/ep | panel (final epoch) | weighted F1 | macro F1 | micro F1 | job |
|---|---|:---:|---|:---:|:---:|:---:|---|
| shuffle, full-exposure (ref) | shuffled | 4.84M | `{2, 12, 83, 435, 480}` | 0.6261 | — | — | Evidence 2b |
| shuffle, root-loss | shuffled | 74,633 | `{2, 38, 225, 435, 480}` | 0.6037 | 0.4240 | 0.6319 | 24950004 |
| intact, root-loss | intact | 74,633 | `{5, 83, 207, 225, 480}` | 0.6100 | 0.4565 | 0.6337 | 24944818 |

Reading it: starving the selector costs ≈0.02 weighted F1 **and** swaps out the key genes — the shuffle
arm loses `{12, 83}`, the intact arm loses `{2, 12}`. Test F1 is comparable across trainmodes (it is scored
on the fixed val/test split; the `train_loss_ce` batch-size caveat affects only epoch-averaged *train* loss).
Two things this is **not**: the +0.006 intact-vs-shuffle root-loss gap is not the clean +0.082 of Evidence 1
— these two runs select *different* panels under a live (starved) selector, so panel quality and selection
dynamics both differ; and there is still no intact-graph, trainmode-0, `select5` baseline in `data/logs/`,
so the intact root-loss panel has no matched full-exposure counterpart.

---

## Evidence 3 — noisy neighbors are neutral, not harmful

A natural objection to Evidence 2: maybe shuffled neighbors *hurt*, and the selector only looks
graph-independent because two effects cancel. They don't.

| frozen panel, no selector | weighted F1 |
|---|---|
| self-edges only (no graph at all) | 0.6321 |
| shuffled graph (pure noise aggregation) | 0.6282 |

Δ = 0.004, well inside the ~0.005 seed sd. Aggregating over random neighbors is worth **nothing** —
neither the regularization benefit that was hypothesized (noise injection into a residual branch, since
`add_self_loops=False` and `res: true`) nor a penalty. So shuffle is a clean ablation of spatial
information, and the 0.6282 → 0.7103 jump is attributable to the graph carrying real signal.

---

## Why earlier writeups concluded the opposite

Every graph-ablation in this project until 2026-07-30 compared **shuffled vs no-graph** — two
configurations that both lack usable spatial information. Naturally they scored the same (0.6282 vs
0.6321), and that was read as "the graph contributes little."

The comparison that was never run was **shuffled vs intact at a matched panel**. The one apparent
intact-graph reference point, `select15` at 0.8533, differs in panel size, so it couldn't be used.

Two phrasings to retire:

- ❌ "The neighborhood adds little" / "the graph is irrelevant." Falsified for classification.
- ❌ "Shuffle scores essentially the same as the intact-graph runs." That compared `select5` shuffle to
  `select15` intact — different panel sizes.

Still correct as written:

- ✅ "GFS's *selection* advantage was never coming from the graph."
- ✅ "The selector finds a better panel with no spatial information at all."

---

## How to write this up

> Gene panel selection does not require spatial context: the selector recovers the same panel with the
> graph destroyed two different ways (5 seeds each), and its advantage over tabular baselines comes from
> mask sample efficiency — patch expansion supplies ~65× more Gumbel mask draws per epoch for free.
> Cell-type classification *does* benefit from spatial context: with the panel frozen and the selector
> removed, restoring the true cell→node assignment improves weighted F1 from 0.628 to 0.710.

Two contributions, cleanly separable, neither of which requires overstating the other. Do not claim the
graph helps *select* genes, and do not claim the graph is unnecessary for the *model*.

---

## Reproducing

```bash
cd scripts
# Evidence 1 — the +0.082 A/B (A100 ~4.5 h and ~8.2 h; prefer --gres=gpu:l40s:1)
sbatch main_panel_graph.sh 0 "[2,12,83,435,480]" panelgraph_shuffle true
sbatch main_panel_graph.sh 0 "[2,12,83,435,480]" panelgraph_intact  false

# Evidence 3 — the no-graph reference bar
sbatch main_panel.sh 0 panel5_shuffle "[2,12,83,435,480]" panel_shuffle

# Evidence 2c — starve the selector of mask draws
sbatch main_shuffle_rootloss.sh 0
```

Recompute metrics from `test_pred.pt` (`{'predictions', 'labels'}`; predictions are logits, so
`argmax(1)`) — do not read them off `f1_side_by_side_with_shuffle_plot.png`, and note
`test_overall_acc` is `MulticlassAccuracy(average="weighted")`, which equals macro-recall, not the ~0.66
micro figure.

**Open runs that would strengthen this document:** `xyz_status=false` on the intact arm (attribution of
the +0.082); seeds 1–4 on both panel-graph arms (error bars — currently one seed each); a random 5-gene
panel as the selection floor.
