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
python -u ../gfs/trainers/antelope.py data.prefix="select15_seed0_tauold_500ep" model.n_select=15 data.rand_seed=0 model.tautype="old" trainer.max_epochs=500
