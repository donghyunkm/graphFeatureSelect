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
# Deep-sample (4-hop) / shallow-loss (2-hop) GFS on abundant-subclass-only data (top6 + other).
# Loss computed only on nodes with a complete 2-hop receptive field (trainmode=2); root-only eval.
python -u ../gfs/trainers/antelope_4hop.py --config-name antelope_abundant_4hop data.prefix="abundant_4hop_seed0" model.n_select=5 data.rand_seed=0
