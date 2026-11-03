#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH --array=0
#SBATCH -e ../outs/error-comms-%A_%a.err
#SBATCH -o ../outs/out-comms-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# Restart of nn_4hop seed0 only. The original array run (25585854_15) was ~2x slower
# than its siblings and its val_f1 oscillated wildly through the anneal (min 0.065,
# max 0.748) instead of converging like seeds 1-3 (~0.84-0.88). Fresh relaunch.
# Cancel the stuck original first:  scancel 25585854_15
# Same config as main_nn_4hop_seeds.sh (4-hop complete-field, trainmode=2, HalfHop OFF,
# n_select=5 baked into antelope_nn_4hop.yaml). See docs/20260907_subclass_grouping_deep_sample.md.

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs

SEED=${SLURM_ARRAY_TASK_ID}
python -u ../gfs/trainers/antelope_4hop.py --config-name antelope_nn_4hop \
    data.prefix="nn_4hop_seed${SEED}_nohh" \
    data.rand_seed=${SEED}
