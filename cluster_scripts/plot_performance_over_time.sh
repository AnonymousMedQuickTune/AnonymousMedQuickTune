#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -J plot_perf_over_time

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
EXPERIMENT_DIR="$BASE_DIR/experiments/$EXPERIMENT_DIR"

cd "$BASE_DIR/code/MedQuickTune"

echo "Starting plot generation for single experiment: $EXPERIMENT_DIR"

if [ -z "$OUTPUT_PATH" ]; then
    python src/analysis/plot_results_over_time.py "$EXPERIMENT_DIR"
else
    OUTPUT_PATH_FULL="$BASE_DIR/$OUTPUT_PATH"
    python src/analysis/plot_results_over_time.py "$EXPERIMENT_DIR" --output "$OUTPUT_PATH_FULL"
fi

echo "Finished"

