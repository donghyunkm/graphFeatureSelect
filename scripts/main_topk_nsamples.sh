#!/bin/bash
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run a single task
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --mem=128g
#SBATCH -p qTRDGPUM
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
#SBATCH --exclude=arctrdagn015


sleep 10s

export PATH=/data/users1/dkim195/miniconda3/bin:$PATH
source /data/users1/dkim195/miniconda3/etc/profile.d/conda.sh
conda activate /data/users1/dkim195/miniconda3/envs/gfs
python ../gfs/trainers/antelope_topk_nsamples.py topk.sampler="pps" topk.pps_sample="pareto" topk.pps_activation="sigmoid_topk" topk.pps_gradient="straight_through" topk.subgraph=False data.prefix="constantlr_tau_1e3to1" trainer.max_epochs=500 topk.num_samples=10 model.tautype="exp"
