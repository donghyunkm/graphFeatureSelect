#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs
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

# deep_sample replication on the 'nn' subclass-grouping variant:
# 4-hop complete-field (trainmode=2), HalfHop OFF, n_select=5 (baked into antelope_nn_4hop.yaml), seeds 0-4.
# Array task id -> rand_seed. See docs/20260907_subclass_grouping_deep_sample.md.

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs

SEED=${SLURM_ARRAY_TASK_ID}
python -u ../gfs/trainers/antelope_4hop.py --config-name antelope_nn_4hop \
    data.prefix="nn_4hop_seed${SEED}_nohh" \
    data.rand_seed=${SEED}
