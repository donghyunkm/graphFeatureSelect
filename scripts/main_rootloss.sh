#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs_root
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH -e ../outs/error-comms-%A.err
#SBATCH -o ../outs/out-comms-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# trainmode=1: loss on ROOT (seed) nodes only -- 64 loss terms/step instead of the whole
# ~4147-node patch. Differs from main.sh in exactly one way.
#
# WHY: the selector's mask-exposure asymmetry. Total gradient *weight* per cell per epoch is
# unchanged (0.0156 both ways -- mean reduction, see CLAUDE.md's retraction), but the number of
# (cell, Gumbel mask) pairs the selection layer is scored on drops 65x:
#   trainmode 0: 4,839,547 loss terms/epoch
#   trainmode 1:    74,633 loss terms/epoch   (1 per cell, as a root)
# For a deterministic classifier those repeats add no information -- which is why upsampling the
# tabular baseline moved it only 0.002. For a *stochastic* selection layer each repeat is a fresh
# mask draw, so they do. This is the one asymmetry vs the frozen-panel baselines that no run has
# held constant.
#
# READ THE PANEL, NOT JUST F1. Compare selections.csv against the trainmode-0 runs, which converge
# on {2, 12, 83, 435, 480} at probs -> 0.999:
#   same panel  -> mask exposure was NOT what found those genes; the gene-quality conclusion stands
#   drifts, or probs stall below 0.99 -> mask exposure WAS the mechanism, and this run is the fair
#                                       comparison to the tabular baselines
#
# Usage:  sbatch main_rootloss.sh
# Prefer l40s:  sbatch --gres=gpu:l40s:1 main_rootloss.sh
# Runtime ~5-7 h/100 ep. NOT a 65x speedup: the forward pass still covers the whole patch, and
# NeighborLoaderMod's subgraph_id loop (96% of step time) is untouched by trainmode.

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
python -u ../gfs/trainers/antelope.py data.prefix="select5_seed0_rootloss" model.n_select=5 model.trainmode=1 model.tautype="old" trainer.max_epochs=100 data.rand_seed=0
