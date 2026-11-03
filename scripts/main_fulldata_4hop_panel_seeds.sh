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

# Feature-ablation arm: 4-hop GFS WITHOUT selection (fs_method=none), frozen to the PERSIST
# select-5 panels regenerated post-label-fix by jobs 25507771-5 (non-upsampled, same full-data
# train/val masks) at ~/persist/results/persist_panel_select5_seed{0-4}.csv, 0-indexed. One run per
# seed's panel, paired with the 4-hop with-selection runs (main_fulldata_4hop_seeds.sh). HalfHop off,
# full 22-label data. See docs/feature_ablation.md. Array task id -> seed -> its own frozen panel.
# NOTE: do NOT revert to persist_genes_selected_genes.csv (stale, pre-label-fix, compromised).

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs

# Array task id selects WHICH PERSIST panel to freeze; the model training seed is fixed to 0 for
# all runs (data.rand_seed=0 below) so init/shuffle noise is not a confound — the only thing that
# differs across the 5 runs is the panel itself.
SEED=${SLURM_ARRAY_TASK_ID}
# PERSIST select-5 panels, one per seed (regenerated 2026-08-31 by jobs 25507771-5;
# from ~/persist/results/persist_panel_select5_seed{0-4}.csv, selected_genes column).
case ${SEED} in
  0) GENES="[2,12,174,435,480]" ;;
  1) GENES="[2,25,83,207,225]" ;;
  2) GENES="[2,83,250,375,480]" ;;
  3) GENES="[2,12,25,115,435]" ;;
  4) GENES="[2,83,86,271,435]" ;;
  *) echo "no panel for seed ${SEED}"; exit 1 ;;
esac

python -u ../gfs/trainers/antelope_4hop_panel.py --config-name antelope_4hop_panel \
    data.genes="${GENES}" \
    data.prefix="full_4hop_panel_panel${SEED}_nohh" \
    data.rand_seed=0 \
    model.halfhop=false
