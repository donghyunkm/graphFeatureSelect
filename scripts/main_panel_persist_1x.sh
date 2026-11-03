#!/bin/bash
#SBATCH -p celltypes
#SBATCH --job-name=gfs_panel1x
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

# Self-edges (no-graph) baseline for the regenerated PERSIST select-5 panels, to get TEST metrics
# for the PERSIST genes and compare them to the GFS full-graph model. See docs/baseline_comparisons.md.
#
# NON-UPSAMPLED (1x): each train cell is seen once. Datasets built with
#   python -m gfs.data.node_exposure --genes <panel> --no-upsample --materialize --emit-valtest \
#       --tag persist_1x_seed<i>
# so exposure/persist_1x_seed{0-4}/ hold a plain 74,633-row train + plain val/test (31,562 each).
#
# rand_seed=0 for ALL runs (user directive): each run already differs only by its PERSIST panel, so
# fixing the seed removes init/shuffle noise as a confound. batch_size=512 (config default 4147 is
# for the ~4.8M-row upsampled set; on 1x it would be ~18 steps/epoch) -> ~146 steps/epoch.
# Panels 0-indexed, from ~/persist/results/persist_panel_select5_seed{0-4}.csv (jobs 25507771-5).

sleep 10s

source /home/donghyun.kim/miniconda3/etc/profile.d/conda.sh
conda activate gfs

PANEL=${SLURM_ARRAY_TASK_ID}
case ${PANEL} in
  0) GENES="[2,12,174,435,480]" ;;
  1) GENES="[2,25,83,207,225]" ;;
  2) GENES="[2,83,250,375,480]" ;;
  3) GENES="[2,12,25,115,435]" ;;
  4) GENES="[2,83,86,271,435]" ;;
  *) echo "no panel for ${PANEL}"; exit 1 ;;
esac

python -u ../gfs/trainers/antelope_panel.py --config-name antelope_panel \
    data.exposure_dir="exposure/persist_1x_seed${PANEL}" \
    data.genes="${GENES}" \
    data.batch_size=512 \
    data.rand_seed=0 \
    data.prefix="persist_1x_panel${PANEL}"
