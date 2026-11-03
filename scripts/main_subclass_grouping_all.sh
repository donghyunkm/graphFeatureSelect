#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH --array=0-29%4
#SBATCH -e ../outs/error-comms-%A_%a.err
#SBATCH -o ../outs/out-comms-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# All 30 runs of the subclass-grouping deep_sample experiment in ONE array, throttled to 4 concurrent
# tasks (--array=0-29%4): the rest queue until a slot frees. 3 datasets x 2 arms x 5 seeds.
# See docs/20260907_subclass_grouping_deep_sample.md.
#
# Task id -> (dataset, arm, seed):
#   dataset = id / 10   (0=glut, 1=nn, 2=gaba)
#   arm     = (id % 10) / 5   (0=2-hop/antelope.py, 1=4-hop/antelope_4hop.py)
#   seed    = id % 5

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs

ID=${SLURM_ARRAY_TASK_ID}
DS=$(( ID / 10 ))
ARM=$(( (ID % 10) / 5 ))
SEED=$(( ID % 5 ))

case ${DS} in
  0) TAG="glut" ;;
  1) TAG="nn" ;;
  2) TAG="gaba" ;;
  *) echo "bad dataset idx ${DS}"; exit 1 ;;
esac

if [[ ${ARM} -eq 0 ]]; then
  TRAINER="antelope.py";      CFG="antelope_${TAG}";       PREFIX="${TAG}_select5_seed${SEED}_nohh"
else
  TRAINER="antelope_4hop.py"; CFG="antelope_${TAG}_4hop";  PREFIX="${TAG}_4hop_seed${SEED}_nohh"
fi

echo "task ${ID} -> dataset=${TAG} arm=${ARM} seed=${SEED} | trainer=${TRAINER} config=${CFG}"
python -u ../gfs/trainers/${TRAINER} --config-name ${CFG} \
    data.prefix="${PREFIX}" \
    data.rand_seed=${SEED}
