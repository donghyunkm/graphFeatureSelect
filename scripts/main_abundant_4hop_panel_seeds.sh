#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --time 60:00:00
#SBATCH --array=0-4
#SBATCH -e ../outs/error-comms-%A_%a.err
#SBATCH -o ../outs/out-comms-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donghyun.kim@alleninstitute.org

# Feature-ablation arm (ABUNDANT 7-label subset): 4-hop GFS WITHOUT selection (fs_method=none),
# frozen to the PERSIST select-5 panels generated ON THE ABUNDANT DATA (persist main.sh with
# data_subdir=0810_abudant_classes_only_top6, seeds 0-4 -> results/persist_panel_select5_seed{0-4}_abundant.csv,
# 0-indexed). One run per seed's panel, compared against the abundant 4-hop with-selection HalfHop-OFF
# runs (docs/20260820_deep_sample_shallow_loss.md, n_select=5, seeds 0-4). HalfHop off, 7-label data.
# See docs/feature_ablation_abundant.md. Array task id -> seed -> its own frozen panel.
# NOTE: do NOT paste the full-data PERSIST panels here — those index the 22-label h5ad.

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs

# Array task id selects WHICH PERSIST panel to freeze; the model training seed is fixed to 0 for
# all runs (data.rand_seed=0 below) so init/shuffle noise is not a confound — the only thing that
# differs across the 5 runs is the panel itself.
SEED=${SLURM_ARRAY_TASK_ID}
# ABUNDANT PERSIST select-5 panels, one per seed, read from
# ~/persist/results/persist_panel_select5_seed{0-4}_abundant.csv (selected_genes column, 0-indexed;
# verified exposure=none, train_rows=74634 non-upsampled). Filled 2026-09-27.
case ${SEED} in
  0) GENES="[2,167,185,205,480]" ;;  # Grik3, Cldn5, Ighm, Unc5d, Dscaml1
  1) GENES="[1,2,83,416,480]" ;;     # Col5a1, Grik3, A830036E02Rik, Adgrf5, Dscaml1
  2) GENES="[2,5,167,205,480]" ;;    # Grik3, Syt6, Cldn5, Unc5d, Dscaml1
  3) GENES="[2,167,185,435,480]" ;;  # Grik3, Cldn5, Ighm, Slc30a3, Dscaml1
  4) GENES="[2,38,167,435,480]" ;;   # Grik3, Rorb, Cldn5, Slc30a3, Dscaml1
  *) echo "no panel for seed ${SEED}"; exit 1 ;;
esac

python -u ../gfs/trainers/antelope_4hop_panel.py --config-name antelope_abundant_4hop_panel \
    data.genes="${GENES}" \
    data.prefix="abundant_4hop_panel_panel${SEED}_nohh" \
    data.rand_seed=0 \
    model.halfhop=false
