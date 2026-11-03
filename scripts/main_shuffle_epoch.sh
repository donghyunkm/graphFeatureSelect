#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH -e ../outs/error-comms-%A.err
#SBATCH -o ../outs/out-comms-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# Seed comes from $1, defaulting to 0:
#   sbatch main_shuffle_epoch.sh 3
# For all five seeds:
#   for s in 0 1 2 3 4; do sbatch main_shuffle_epoch.sh $s; done
SEED=${1:-0}

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
python -u ../gfs/trainers/antelope_shuffle_epoch.py \
    data.prefix="select5_seed${SEED}_shuffleepoch" \
    model.n_select=5 \
    data.rand_seed=${SEED}
