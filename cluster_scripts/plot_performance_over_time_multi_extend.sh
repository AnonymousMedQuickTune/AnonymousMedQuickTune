#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -J plot_perf_over_time_multi_extend

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"

cd "$BASE_DIR/code/MedQuickTune"

echo "Starting plot generation for multiple experiments with extend flag"
echo "Plot name: $PLOT_NAME"
echo "Experiment directories: $EXPERIMENT_DIRS"
echo "Output path: $OUTPUT_PATH"

OUTPUT_PATH_FULL="$BASE_DIR/$OUTPUT_PATH"
OUTPUT_DIR=$(dirname "$OUTPUT_PATH_FULL")
mkdir -p "${OUTPUT_DIR}"

# Convert EXPERIMENT_DIRS to full paths
EXPERIMENT_DIRS_FULL=""
for DIR in $EXPERIMENT_DIRS; do
    EXPERIMENT_DIRS_FULL="$EXPERIMENT_DIRS_FULL $BASE_DIR/experiments/$DIR"
done

python src/analysis/plot_results_over_time.py $EXPERIMENT_DIRS_FULL --output "$OUTPUT_PATH_FULL" --extend-to-max-configs

echo "Finished"

