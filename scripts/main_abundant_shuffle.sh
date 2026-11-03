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


sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
# Shuffled-cell control on the SAME abundant-subclass-only graph:
# gene expression + labels permuted together, spatial coords + graph edges left intact
python -u ../gfs/trainers/antelope_shuffle.py --config-name antelope_abundant data.prefix="abundant_select5_seed0_shuffle" model.n_select=5 data.rand_seed=0
