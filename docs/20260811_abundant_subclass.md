# 0811 — Abundant-subclass: intact graph vs shuffled-cell control (n_select=5, seed 0)

**Question.** On the abundant-subclass-only data (7 labels: top-6 abundant + `other`, 500 genes),
does neighborhood/graph information change *which gene panel gets selected*? The shuffle arm is the
matched graph-ablation control (one shared permutation of gene expr + labels; xyz + edges intact).

**Runs** (both completed overnight, 100 epochs, `n_select=5`, `seed=0`):
- default → `data/logs/20260810_204705_abundant_select5_seed0_s0/`
- shuffle → `data/logs/20260810_204836_abundant_select5_seed0_shuffle_s0/`

Metrics recomputed from each run's `test_pred.pt` (not read off any plot); n = 31,563 test cells.

## Test metrics

| Metric            | default (intact) | shuffle (control) | Δ (intact − shuffle) |
|-------------------|:---------------:|:-----------------:|:--------------------:|
| **Weighted F1**   | **0.7789**      | 0.7365            | **+0.0424**          |
| Macro F1          | 0.7618          | 0.7106            | +0.0512              |
| Accuracy          | 0.7810          | 0.7399            | +0.0411              |
| Balanced accuracy | 0.7616          | 0.7089            | +0.0527              |

The intact graph is worth **+0.042 weighted / +0.051 macro F1** here — the same *direction* as the
full-isocortex frozen-panel result (+0.082, see `MEMORY.md → panel-graph-arm-noise-robustness`),
though smaller on this easier 7-class subset. This is the **classification** benefit of the graph,
not a selection effect (see the panel below).

**Bootstrap significance (cell-level).** This is a **single-seed** comparison (n=1 run per arm), so a
seed-level paired test is unavailable — instead bootstrap the 31,563 test cells (B=20 000, resample cells,
recompute each arm's F1, take the difference). Δ = intact − shuffle:

| metric | Δ | 95% CI | bootstrap 2-sided p | P(Δ>0) |
|---|---|---|---|---|
| weighted | +0.0424 | [+0.0356, +0.0491] | <0.001 | 1.000 |
| macro | +0.0511 | [+0.0436, +0.0588] | <0.001 | 1.000 |

Both CIs are comfortably above zero (no resample crossed it) — the intact-graph advantage is real on this run's
test set. ⚠️ scope: this bootstraps **test-cell sampling variability for one seed/split only**; it is not a
seed-level claim (contrast the paired 4-hop-vs-2-hop docs, which vary the seed). The shuffle permutes labels, so
the two arms are not per-cell paired — indices are resampled over the shared test population, ~unpaired.

### Per-class F1 (label → subclass, LabelEncoder = sorted categories)

| label | subclass              | test n | default F1 | shuffle F1 | Δ      |
|:-----:|-----------------------|:------:|:----------:|:----------:|:------:|
| 0     | 004 L6 IT CTX Glut    | 2058   | 0.577      | 0.422      | +0.155 |
| 1     | 006 L4/5 IT CTX Glut  | 5693   | 0.882      | 0.853      | +0.029 |
| 2     | 007 L2/3 IT CTX Glut  | 3063   | 0.839      | 0.761      | +0.078 |
| 3     | 030 L6 CT CTX Glut    | 3819   | 0.784      | 0.757      | +0.027 |
| 4     | 319 Astro-TE NN       | 3491   | 0.722      | 0.712      | +0.010 |
| 5     | 333 Endo NN           | 2522   | 0.761      | 0.742      | +0.019 |
| 6     | other                 | 10917  | 0.767      | 0.727      | +0.040 |

The graph helps most on the hardest/rarest IT-glutamatergic class (004 L6 IT, +0.155) — consistent
with neighborhood context disambiguating layer-specific excitatory types that are hard to tell apart
from expression alone.

## Selected gene panels

**The two arms select the identical 5-gene panel** (final epoch; both had converged with selection
probabilities ≈ 0.99 by the last epochs):

| arm     | selected indices (final epoch) | as a set              |
|---------|--------------------------------|-----------------------|
| default | 12, 38, 2, 480, 83             | **{2, 12, 38, 83, 480}** |
| shuffle | 2, 38, 12, 480, 83             | **{2, 12, 38, 83, 480}** |

Gene index → Ensembl ID:

| index | Ensembl ID          |
|:-----:|---------------------|
| 2     | ENSMUSG00000001985  |
| 12    | ENSMUSG00000033278  |
| 38    | ENSMUSG00000036192  |
| 83    | ENSMUSG00000084890  |
| 480   | ENSMUSG00000032087  |

## Takeaway

- **Selection does NOT depend on the graph** — even restricted to abundant subclasses, destroying the
  neighborhood signal leaves the selected panel *exactly* the same `{2, 12, 38, 83, 480}`. This extends
  the earlier full-isocortex finding to the abundant-subclass regime.
- **Classification DOES benefit from the graph** — same panel, intact vs shuffled, +0.042 weighted /
  +0.051 macro F1, concentrated on the hardest excitatory type.

## Next steps?

In all our current experiments, selection is averaged out over hundreds of thousands of `(cell, mask)` Gumbel draws. Suggestions below pushes into a regime where that averaging is weaker or the graph signal is stronger, so the intact-vs-shuffle panels *could* diverge.

- **Smaller / collaborator-relevant brain region.** Restrict to the specific region our collaborator
  cares about. Beyond serving the actual deliverable, a region has far fewer training cells → far
  fewer Gumbel mask draws per epoch → selection is *less averaged*, so a weak neighborhood signal that
  gets washed out at full scale could finally tip which gene wins.

- **Degrade the expression signal.** Introduce noise to the cell's gene expression to simulate type of noise found in budget-limited transcriptomics experiments?  If expression alone stops being sufficient to pick genes, the selector may lean on neighborhood structure — and the intact vs shuffle panels could split.