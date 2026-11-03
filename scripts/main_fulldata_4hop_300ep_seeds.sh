#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH --array=0-1
#SBATCH -e ../outs/error-comms-%A_%a.err
#SBATCH -o ../outs/out-comms-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org


sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs
# 300-epoch variant of the 4-hop (deep-sample / shallow-loss) arm (4hop-vs-2hop comparison,
# FULL 22-label data). Identical to main_fulldata_4hop_seeds.sh but trainer.max_epochs=300:
# the tautype="old" schedule stretches with max_epochs, so 300 epochs gives ~3x the exploratory
# high-tau steps before selection commits. NOTE: 4-hop is ~12h/seed at 100ep, so expect ~36h/seed
# (within the 60h walltime). Probe run: 2 seeds only (array 0-1).
# See docs/4hop_2hop_comparison.md. Array task id -> rand_seed.
SEED=${SLURM_ARRAY_TASK_ID}
python -u ../gfs/trainers/antelope_4hop.py --config-name antelope_4hop \
    data.prefix="full_4hop300_seed${SEED}_nohh" \
    model.n_select=5 \
    data.rand_seed=${SEED} \
    model.halfhop=false \
    trainer.max_epochs=300
