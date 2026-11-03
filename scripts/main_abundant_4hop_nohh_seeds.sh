#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH --array=1-4
#SBATCH -e ../outs/error-comms-%A_%a.err
#SBATCH -o ../outs/out-comms-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org


sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
# Seeds 1-4 of the deep-sample (4-hop) / shallow-loss (trainmode=2) GFS with HalfHop OFF (train + eval).
# Error-bar companion to main_abundant_4hop_nohh.sh (seed 0); array task id -> rand_seed.
SEED=${SLURM_ARRAY_TASK_ID}
python -u ../gfs/trainers/antelope_4hop.py --config-name antelope_abundant_4hop data.prefix="abundant_4hop_seed${SEED}_nohh" model.n_select=5 data.rand_seed=${SEED} model.halfhop=false
