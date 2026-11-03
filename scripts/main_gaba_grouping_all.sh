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

# Gaba fine-grained grouping experiment (2-hop vs 4-hop deep_sample) in ONE array, throttled to 4
# concurrent tasks (--array=0-29%4): the rest queue until a slot frees. 3 datasets x 2 arms x 5 seeds.
# Datasets built from data/0914_all_subclasses/ (enriched 34-label set) via build_subclass_variant.py.
# See docs/20260914_subclass_grouping_finegrained.md.
#
# Task id -> (dataset, arm, seed):
#   dataset = id / 10   (0=gaba_fine [9 lbl], 1=gaba_cge [4 lbl], 2=gaba_mge [6 lbl])
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
  0) TAG="fine" ;;
  1) TAG="cge" ;;
  2) TAG="mge" ;;
  *) echo "bad dataset idx ${DS}"; exit 1 ;;
esac

if [[ ${ARM} -eq 0 ]]; then
  TRAINER="antelope.py";      CFG="antelope_gaba_${TAG}";       PREFIX="gaba_${TAG}_seed${SEED}_nohh"
else
  TRAINER="antelope_4hop.py"; CFG="antelope_gaba_${TAG}_4hop";  PREFIX="gaba_${TAG}_4hop_seed${SEED}_nohh"
fi

echo "task ${ID} -> dataset=gaba_${TAG} arm=${ARM} seed=${SEED} | trainer=${TRAINER} config=${CFG}"
python -u ../gfs/trainers/${TRAINER} --config-name ${CFG} \
    data.prefix="${PREFIX}" \
    data.rand_seed=${SEED}
