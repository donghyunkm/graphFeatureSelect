#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs_panel
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH -e ../outs/error-comms-%A.err
#SBATCH -o ../outs/out-comms-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# Frozen-gene-panel baseline: no graph (self-edges), no selection module.
# Trains on the upsampled tabular dataset so gradient updates match the graph run
# (batch 4147 -> 1167 steps/epoch, same 1/4147 per-row weight in the batch mean).
#
# Usage:  sbatch main_panel.sh [seed] [exposure_dir] [genes] [prefix]
#   sbatch main_panel.sh                       # persist s0 panel, seed 0
#   sbatch main_panel.sh 1                     # same panel, seed 1
#   for s in 0 1 2 3 4; do sbatch main_panel.sh $s; done
#
# For a different panel, build the dataset first, then pass dir + genes:
#   python -m gfs.data.node_exposure --epochs 5 --genes 2,12,83,435,480 \
#       --materialize --emit-valtest --tag shuffle_select5
#   sbatch main_panel.sh 0 exposure/shuffle_select5 "[2,12,83,435,480]" panel_shuffle

sleep 10s

SEED=${1:-0}
EXPOSURE_DIR=${2:-"exposure/persist_seed0_select5"}
GENES=${3:-"[2,250,435,480,481]"}
PREFIX=${4:-"panel_persist_s0panel"}

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
python -u ../gfs/trainers/antelope_panel.py \
    data.exposure_dir="${EXPOSURE_DIR}" \
    data.genes="${GENES}" \
    data.prefix="${PREFIX}" \
    data.rand_seed=${SEED}
