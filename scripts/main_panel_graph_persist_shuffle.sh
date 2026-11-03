#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs_pgshuf
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH --array=0-4
#SBATCH -e ../outs/error-comms-%A_%a.err
#SBATCH -o ../outs/out-comms-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# baseline_comparisons.md, SHUFFLED-GRAPH arm: PERSIST select-5 panel + GFS model over a SHUFFLED
# graph (shuffle_cells=true). The graph carries no real neighbour signal (cells permuted, expression
# <-> label pairing kept), but the 2-hop NeighborLoader still gives each cell the graph's ~65 loss
# terms/epoch by construction -- i.e. this is the "~65x exposure + noisy conv, no real neighbours"
# arm, contrasting the self-edges 1x baseline (no graph, no exposure).
#
# HalfHop OFF and xyz ON to MATCH the GFS full-graph reference (job 25507587, antelope.yaml) this doc
# compares against, and to avoid the eval-time HalfHop perturbation (MEMORY.md eval-time-halfhop-active).
# NOTE: this differs from the older panel_graph shuffle runs (memory panel-graph-arm-noise-robustness),
# which used the config default halfhop=true -- so do not cross-compare F1 with those.
#
# rand_seed=0 for ALL runs: the panel is the only thing that varies. (rand_seed also seeds the cell
# permutation, so the shuffled graph is identical across the 5 runs -- a fixed graph ablation.)
# Array task id -> panel index. Panels 0-indexed, from ~/persist/results (jobs 25507771-5).
# Runtime: NeighborLoaderMod's subgraph_id loop dominates (~7-18 h/100 ep); prefer l40s if free.

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs

PANEL=${SLURM_ARRAY_TASK_ID}
case ${PANEL} in
  0) GENES="[2,12,174,435,480]" ;;
  1) GENES="[2,25,83,207,225]" ;;
  2) GENES="[2,83,250,375,480]" ;;
  3) GENES="[2,12,25,115,435]" ;;
  4) GENES="[2,83,86,271,435]" ;;
  *) echo "no panel for ${PANEL}"; exit 1 ;;
esac

python -u ../gfs/trainers/antelope_panel_graph.py --config-name antelope_panel_graph \
    data.genes="${GENES}" \
    data.shuffle_cells=true \
    data.rand_seed=0 \
    data.prefix="persist_shufgraph_panel${PANEL}" \
    model.halfhop=false
