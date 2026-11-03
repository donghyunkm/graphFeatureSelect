#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs_shufroot
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH -e ../outs/error-comms-%A.err
#SBATCH -o ../outs/out-comms-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# main_rootloss.sh but with the CELLS SHUFFLED, at model.trainmode=1.
#
# Uses antelope_shuffle_within.py, not antelope_shuffle.py. The two differ ONLY in the valtest
# permutation: the original permutes valtest cells across the whole 63,124-cell file, which is a
# bijection (no leak -- val and test partition the file, so val nodes get 31,562 distinct cells and
# test nodes get exactly the complement) but re-partitions WHICH cells constitute each split,
# swapping about half of each. hemisphere_new_shuffle_within.py permutes within val and within test
# separately, so both splits hold exactly the cells the h5ad assigns them while the spatial
# arrangement inside each is still destroyed. train is byte-identical to the original (verified).
# val_mask/test_mask, xyz and edge_index are never permuted in either.
#
# The 2x2 this completes. Rows = mask exposure (trainmode), cols = graph:
#
#                          intact graph                shuffled cells
#   trainmode 0  (4.84M)   main.sh (no select5 run!)   main_shuffle.sh   -> 0.627, {2,12,83,435,480}
#   trainmode 1  (74,633)  main_rootloss.sh            THIS
#
# WHY the shuffled version matters: the 5 completed shuffle runs are the arm that produced the
# headline panel {2, 12, 83, 435, 480} -- the one the frozen-panel run scored at 0.6321 against
# Persist's 0.5748. So the shuffle arm, not the intact arm, is where "how did the selector find
# these genes" actually needs answering, and it is the only trainmode-0 select5 baseline that
# exists (there is no intact-graph select5 trainmode-0 run in data/logs/). Against those 5 seeds
# this is a strictly one-variable comparison; main_rootloss.sh is not, until the intact
# trainmode-0 select5 control is run.
#
# READ THE PANEL, NOT JUST F1. Compare selections.csv (last epoch) against the shuffle seeds,
# which land on {2, 12, 83, 435, 480} at probs -> 0.999 (4-5 of 5 picks, 23/25 across seeds):
#   same panel -> mask exposure was NOT the mechanism; the selector is sample-efficient and the
#                 gene-quality conclusion needs no qualification
#   drifts, or probs stall below ~0.99 -> mask exposure WAS the mechanism, and THIS run (not the
#                 trainmode-0 one) is the fair comparison against the tabular baselines
# Don't read a verdict off the first ~10 epochs: every trainmode-0 select5/select15 run also sits
# on {393, 38, 472, 280, 228} that early and only migrates later in annealing.
#
# The within-split change makes F1 comparison to those 5 seeds slightly non-strict (they used the
# cross-split permutation). Expected to be tiny -- val and test are statistically interchangeable
# here, subclass distributions agreeing to 4e-4 L1 and xyz means to 3 decimals -- but the PANEL,
# which is the actual readout, is set by training and is unaffected either way.
#
# tautype is pinned "old" explicitly: tau_schedule falls through to the two-stage schedule for
# ANY value that is not the literal string "old", and main_shuffle.sh never passed it. The
# completed shuffle runs inherited antelope.yaml's value -- verified "old" in
# data/logs/20260721_180328_select5_seed0_shuffle_s0/lightning_logs/version_0/hparams.yaml, so
# passing it here keeps the comparison one-variable rather than changing it.
#
# Usage:  sbatch main_shuffle_rootloss.sh [seed]     (default 0)
# Prefer l40s:  sbatch --gres=gpu:l40s:1 main_shuffle_rootloss.sh 0
# Runtime ~5-7 h/100 ep. NOT a 65x speedup: trainmode only changes which rows are indexed for
# the loss. The forward pass still covers the whole ~4147-node patch and NeighborLoaderMod's
# subgraph_id python loop (96% of step time) is untouched.
#
# Note batch_size is logged as train_mask.sum() (~4147) regardless of trainmode
# (antelope.py:395), so epoch-averaged train_loss_ce is not comparable across trainmodes.

SEED=${1:-0}

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
python -u ../gfs/trainers/antelope_shuffle_within.py \
    data.prefix="select5_seed${SEED}_shufflerootloss" \
    model.n_select=5 \
    model.trainmode=1 \
    model.tautype="old" \
    trainer.max_epochs=100 \
    data.rand_seed=${SEED}
