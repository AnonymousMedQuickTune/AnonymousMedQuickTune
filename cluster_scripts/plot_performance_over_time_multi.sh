#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -J plot_perf_over_time_multi

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
OUTPUT_DIR="experiments/Plots"
OUTPUT_PATH="${OUTPUT_DIR}/${PLOT_NAME}.png"

cd "$BASE_DIR/code/MedQuickTune"

OUTPUT_PATH_FULL="$BASE_DIR/$OUTPUT_PATH"
OUTPUT_DIR_FULL=$(dirname "$OUTPUT_PATH_FULL")
mkdir -p "${OUTPUT_DIR_FULL}"

# Convert EXPERIMENT_DIRS to full paths
# Handle both relative paths (without /) and absolute paths
EXPERIMENT_DIRS_FULL=""
for DIR in $EXPERIMENT_DIRS; do
    if [[ "$DIR" == /* ]]; then
        # Absolute path, use as is
        EXPERIMENT_DIRS_FULL="$EXPERIMENT_DIRS_FULL $DIR"
    else
        # Relative path, prepend BASE_DIR/experiments/
        EXPERIMENT_DIRS_FULL="$EXPERIMENT_DIRS_FULL $BASE_DIR/experiments/$DIR"
    fi
done

python src/analysis/plot_results_over_time.py $EXPERIMENT_DIRS_FULL --output "$OUTPUT_PATH_FULL"

