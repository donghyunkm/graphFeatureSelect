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
# GR2ST attention-entropy comparison: two 4-hop, 100-epoch, select-5, raw-graph (HalfHop OFF) runs on the
# FULL 22-label data. The ONLY difference between the two arms is the attention-entropy loss weight; SAME
# seed (rand_seed=0) so data split + init are identical -> a clean paired on/off comparison.
#   array task 0 = baseline  (attention_entropy_weight=0.0, but log_attention_stats=true so it still logs
#                             the entropy/neighbourhood diagnostics for comparison)
#   array task 1 = GR2ST     (attention_entropy_weight=0.01, the GR2ST default; diagnostics logged automatically)
# Diagnostics on both arms: train/val attention_entropy, top1_mass, effective_neighbors, normalized_entropy.
# Checkpoint metric is unchanged (val_f1_overall, max). See docs/20260902_gr2st_attention_entropy.md.
SEED=0
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    AE_WEIGHT=0.0
    PREFIX="full_4hop100_ae0_nohh"
else
    AE_WEIGHT=0.01
    PREFIX="full_4hop100_ae0p01_nohh"
fi

python -u ../gfs/trainers/antelope_4hop.py --config-name antelope_4hop \
    data.prefix="${PREFIX}" \
    model.n_select=5 \
    data.rand_seed=${SEED} \
    model.halfhop=false \
    trainer.max_epochs=100 \
    model.attention_entropy_weight=${AE_WEIGHT} \
    model.log_attention_stats=true
