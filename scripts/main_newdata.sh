#!/bin/bash
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run a single task
#SBATCH --cpus-per-task=5 # Number of CPU cores per task
#SBATCH --mem=128g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 7000
#SBATCH -J comms
#SBATCH -e ../error/error-comms-%A.err
#SBATCH -o ../out/out-comms-%A.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dkim195@gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdagn007

sleep 10s

export PATH=/data/users1/dkim195/miniconda3/bin:$PATH
source /data/users1/dkim195/miniconda3/etc/profile.d/conda.sh
conda activate /data/users1/dkim195/miniconda3/envs/gfs
python ../gfs/trainers/antelope.py data.prefix="gfs_zhuang" data.file_names=["Zhuang-ABCA-1-section80.h5ad"] data.n_genes=1122 data.n_labels=138
sleep 30s
