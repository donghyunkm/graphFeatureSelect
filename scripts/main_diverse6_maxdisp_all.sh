#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH --array=0-9%4
#SBATCH -e ../outs/error-comms-%A_%a.err
#SBATCH -o ../outs/out-comms-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# diverse-6 MAX-DISPERSION abundant-style experiment (2-hop vs 4-hop deep_sample) in ONE array, throttled to
# 4 concurrent tasks (--array=0-9%4). 1 dataset x 2 arms x 5 seeds.
# Dataset = the 6 of the top-12-by-abundance subclasses with maximally spread expression profiles
# (007 L2/3 IT Glut, 052 Pvalb, 053 Sst Gaba, 319 Astro, 327 Oligo, 333 Endo NN) + 'other' (7 labels),
# built from the default 22-label full data via build_subclass_variant.py. See docs/20260919_diverse6_maxdisp.md.
#
# Task id -> (arm, seed):
#   arm  = id / 5    (0=2-hop/antelope.py, 1=4-hop/antelope_4hop.py)
#   seed = id % 5

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs

ID=${SLURM_ARRAY_TASK_ID}
ARM=$(( ID / 5 ))
SEED=$(( ID % 5 ))

if [[ ${ARM} -eq 0 ]]; then
  TRAINER="antelope.py";      CFG="antelope_diverse6_maxdisp";       PREFIX="diverse6_maxdisp_seed${SEED}_nohh"
else
  TRAINER="antelope_4hop.py"; CFG="antelope_diverse6_maxdisp_4hop";  PREFIX="diverse6_maxdisp_4hop_seed${SEED}_nohh"
fi

echo "task ${ID} -> arm=${ARM} seed=${SEED} | trainer=${TRAINER} config=${CFG}"
python -u ../gfs/trainers/${TRAINER} --config-name ${CFG} \
    data.prefix="${PREFIX}" \
    data.rand_seed=${SEED}
