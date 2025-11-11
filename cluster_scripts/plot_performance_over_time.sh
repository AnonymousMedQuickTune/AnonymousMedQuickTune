#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -J plot_perf_over_time

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
# Handle both relative paths (without /) and absolute paths
if [[ "$EXPERIMENT_DIR" == /* ]]; then
    # Absolute path, use as is
    EXPERIMENT_DIR_FULL="$EXPERIMENT_DIR"
else
    # Relative path, prepend BASE_DIR/experiments/
    EXPERIMENT_DIR_FULL="$BASE_DIR/experiments/$EXPERIMENT_DIR"
fi

cd "$BASE_DIR/code/MedQuickTune"

if [ -z "$OUTPUT_PATH" ]; then
    python src/analysis/plot_results_over_time.py "$EXPERIMENT_DIR_FULL"
else
    # Handle both relative and absolute output paths
    if [[ "$OUTPUT_PATH" == /* ]]; then
        OUTPUT_PATH_FULL="$OUTPUT_PATH"
    else
        OUTPUT_PATH_FULL="$BASE_DIR/$OUTPUT_PATH"
    fi
    python src/analysis/plot_results_over_time.py "$EXPERIMENT_DIR_FULL" --output "$OUTPUT_PATH_FULL"
fi

