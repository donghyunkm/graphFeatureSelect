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
# Deep-sample (4-hop) / shallow-loss (2-hop, trainmode=2) GFS with HalfHop OFF (train + eval).
# No-HalfHop comparator to main_abundant_4hop.sh; clean raw-graph eval, full 2-hop fields on loss nodes.
python -u ../gfs/trainers/antelope_4hop.py --config-name antelope_abundant_4hop data.prefix="abundant_4hop_seed0_nohh" model.n_select=5 data.rand_seed=0 model.halfhop=false
