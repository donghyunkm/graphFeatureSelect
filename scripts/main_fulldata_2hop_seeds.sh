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


sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
# 2-hop baseline arm of the 4hop-vs-2hop comparison on the FULL 22-label data.
# GFS select-5, trainmode=0 (loss on all patch nodes), HalfHop OFF (train + eval), 5 seeds.
# See docs/4hop_2hop_comparison.md. Array task id -> rand_seed.
SEED=${SLURM_ARRAY_TASK_ID}
python -u ../gfs/trainers/antelope.py \
    data.prefix="full_2hop_seed${SEED}_nohh" \
    model.n_select=5 \
    data.rand_seed=${SEED} \
    model.halfhop=false
