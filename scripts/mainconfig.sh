#!/bin/bash
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run a single task
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --mem=128g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:1
#SBATCH -t 7000
#SBATCH -J comms
#SBATCH -e ../error/error-comms-%A.err
#SBATCH -o ../out/out-comms-%A.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dkim195@gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdagn007
#SBATCH --array=1-50

sleep 10s

export PATH=/data/users1/dkim195/miniconda3/bin:$PATH
source /data/users1/dkim195/miniconda3/etc/profile.d/conda.sh
conda activate /data/users1/dkim195/miniconda3/envs/gfs

# Specify the path to the config file
config=/data/users1/dkim195/graphFeatureSelect/scripts/config.txt

seed=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
fold=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

python ../gfs/trainers/antelope.py data.rand_seed=${seed} data.cv=${fold} model.n_select=20 data.prefix="0314_20_"

sleep 30s
