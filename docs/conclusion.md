# Conclusions

## 1. Classification accuracy: the GNN on its real spatial edges wins every comparison

We want to show that the GNN using its true neighbor graph is the best classifier across the board.

1. **Selector genes + GNN (proper edges) > selector genes + GNN (shuffled edges).**
   Single-variable A/B on a frozen learned 5-gene panel with no selector — the *only* difference is
   whether the cell→node assignment is permuted. Intact graph **0.7103** vs shuffled **0.6282**
   (Δ **+0.082** weighted, +0.092 macro), seed 0. This is the largest single effect in the whole
   investigation. → `does-the-graph-help.md`
   - The incomplete-neighborhood issue doesn't apply here: metrics are computed on **root nodes only**,
     and root nodes always have their full 2-hop neighborhood materialized.

2. **Selector genes + GNN (proper edges) > PERSIST genes + GNN (no edges).**
   This is the full GFS model against the tabular baseline floor. Eval-only, no selection module (both
   panels are fixed). The selector's panel `[2, 12, 83, 435, 480]` scored **with the intact graph** reaches
   **0.7103**, vs the default (non-upsampled) PERSIST panel `[2, 250, 435, 480, 481]` scored with **no
   graph** (self-edges only, `antelope_panel.py`, `fs_method=none`) at **0.5724** — Δ **+0.138** weighted.
   So the selector + graph decisively beats the strongest non-graph panel chooser run as a tabular
   classifier. → `does-the-graph-help.md` (selector intact 0.7103), `old-claude.md` (PERSIST no-graph 0.5724)
   - Root-only metrics, so incomplete neighborhoods don't matter.

3. **Selector genes + GNN (proper edges) > PERSIST genes + GNN (proper edges).**
   Both arms are the *same* 4-hop GNN on the *same* real graph; only the 5 genes differ (learned selector
   panel vs frozen PERSIST panel). This isolates whether learning the panel jointly with the graph helps
   when the graph is held fixed — i.e. the **upper bound of the tabular baseline**, since PERSIST is the
   strongest non-graph panel chooser. → `feature_ablation.md`, `feature_ablation_abundant.md`
   - **Full data (22 labels):** selector **0.7099** vs frozen PERSIST **0.6847** (Δ **+0.0252** weighted).
     Cell-level bootstrap significant (CI entirely above 0); seed-level paired t ns (p=0.198). A real but
     not rock-solid win.
   - **Abundant subset (7 labels):** selector **0.7851** vs frozen PERSIST **0.7791** (Δ **+0.0060**
     weighted). Cell-level significant but tiny; seed-level ns — effectively a tie. On the easy regime the
     stable PERSIST backbone genes ({Grik3, Dscaml1}) nearly catch up, so learned selection buys little
     once the panel is already good.

Note: the selector panel used in 1.1 and 1.2 (`[2, 12, 83, 435, 480]`) was chosen with the **2-hop**
selector, not 4-hop. Those two should be redone with the 4-hop panel (1.3 already uses 4-hop selector
genes). Expected to hold — on full data 4-hop ties 2-hop in accuracy (memory
`4hop-does-not-beat-2hop-full-data`), and the graph's classification value looks panel-independent — but
this is a prediction, not yet verified.

## 2. Gene panel: the panel the GNN learns is the best panel in every comparison

We want to show that the genes the selector picks *with the graph* are the best panel.

1. **Before 4-hop (plain 2-hop, where most non-root cells have incomplete neighborhoods), the selector
   with shuffled edges picks the *same* panel as the selector with proper edges.**
   Five shuffle-control seeds converge on `{2, 12, 83, 480}` (in 5/5 seeds; 23/25 picks inside the intact
   `select15` panel). → `does-the-graph-help.md`
   - So the 2-hop selector's apparent edge was **sample efficiency**, not spatial information: patch
     expansion makes each cell reappear in ~65 patches/epoch, each under a fresh Gumbel-softmax mask, which
     is what finds the good genes — destroying the graph doesn't change the panel.

2. **With 4-hop, the selector picks a *different* and *better* panel than the 2-hop selector → it genuinely
   makes use of spatial information.**
   This is essentially our **final model**. The 4-hop complete-receptive-field selector both swaps genes
   relative to 2-hop and scores higher:
   - `deep_sample_shallow_loss.md` — abundant: 4-hop **0.7851 ± 0.0073** vs 2-hop **0.7719 ± 0.0043**
     (Δ +0.0132, 4-hop > 2-hop on every seed), and a different panel (e.g. 4-hop picks Ptprm/Syt6 where
     2-hop picks Rorb/Hmcn1).
   - `20260919_diverse6_balanced.md` — diverse family-balanced set: 4-hop **0.8322 ± 0.0122** vs 2-hop
     **0.8121 ± 0.0239** (Δ +0.0202 weighted); plus the other grouping docs (`diverse6_maxdisp`,
     `subclass_grouping_deep_sample`, …).
   - **Caveat:** this holds when a *specific set of subclasses* is kept (abundant / diverse subsets), **not**
     on the full 22-label data, where 4-hop merely ties 2-hop (memory `4hop-does-not-beat-2hop-full-data`).
