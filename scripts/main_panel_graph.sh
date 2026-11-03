#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs_pgraph
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH -e ../outs/error-comms-%A.err
#SBATCH -o ../outs/out-comms-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# Frozen gene panel WITH the graph kept. Differs from main_shuffle.sh in exactly one way:
# fs_method="none", so no selector, no soft-mask annealing, no Gumbel gene-dropout.
# Isolates "noisy neighbours regularize the representations" from "annealing/co-adaptation".
#
# Reading the result vs shuffle GNN 0.627 and tabular panel ~0.57:
#   ~0.62 -> the noisy conv path was doing the work (noise-robustness)
#   ~0.57 -> the conv path contributes nothing; annealing/co-adaptation owns the gap
#
# Usage:  sbatch main_panel_graph.sh [seed] [genes] [prefix] [shuffle_cells]
#   sbatch main_panel_graph.sh                                  # shuffle panel, shuffled cells
#   sbatch main_panel_graph.sh 0 "[2,12,83,435,480]" panelgraph_intact false   # intact-graph control
#   for s in 0 1 2; do sbatch main_panel_graph.sh $s; done
#
# Runtime ~5-18 h/100 epochs; NeighborLoaderMod's subgraph_id loop dominates (~96% of step
# time), same as the shuffle arm which took 7.4 h. Prefer l40s: sbatch --gres=gpu:l40s:1 ...

sleep 10s

SEED=${1:-0}
GENES=${2:-"[2,12,83,435,480]"}
PREFIX=${3:-"panelgraph_shuffle"}
SHUFFLE_CELLS=${4:-"true"}

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
python -u ../gfs/trainers/antelope_panel_graph.py \
    data.genes="${GENES}" \
    data.prefix="${PREFIX}" \
    data.shuffle_cells=${SHUFFLE_CELLS} \
    data.rand_seed=${SEED}
