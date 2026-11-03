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
# GR2ST attention-entropy, fourth weight point: 4-hop, 100-epoch, select-5, raw-graph (HalfHop OFF) on the
# FULL 22-label data. Identical to main_4hop_gr2st_compare.sh except attention_entropy_weight=0.05 (5x
# stronger than the GR2ST default 0.01). SAME seed (rand_seed=0) so data split + init match the other arms
# -> extends the paired sweep to {0.0, 0.001, 0.01, 0.05}. See docs/20260902_gr2st_attention_entropy.md.
SEED=0
AE_WEIGHT=0.05
PREFIX="full_4hop100_ae0p05_nohh"

python -u ../gfs/trainers/antelope_4hop.py --config-name antelope_4hop \
    data.prefix="${PREFIX}" \
    model.n_select=5 \
    data.rand_seed=${SEED} \
    model.halfhop=false \
    trainer.max_epochs=100 \
    model.attention_entropy_weight=${AE_WEIGHT} \
    model.log_attention_stats=true
