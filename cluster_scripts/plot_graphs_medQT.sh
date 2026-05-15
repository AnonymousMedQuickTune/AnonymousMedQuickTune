#!/bin/bash
##SBATCH -p gpu_a100
#SBATCH -p rome
##SBATCH --gres=gpu:1
#SBATCH -J plot_auc_time
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out
#SBATCH --error=/projects/code/MedQuickTune/logs/%x-%j.err

source /medquicktune/bin/activate

BASE_DIR="projects/code/MedQuickTune"

DATASET="lipo"

BASELINE_EXP="densenet"
NEPS_EXP="bo"
QUICKTUNE_EXP="medquicktune"

PLOT_NAME="${DATASET}_auc_comparison"
TITLE="WORC Lipo Dataset"

# 24 hours
COST_TO_SPEND=86400

OUTPUT_DIR="$BASE_DIR/experiments/Plots"
mkdir -p "$OUTPUT_DIR"

cd "$BASE_DIR/code/MedQuickTune"

echo "======================================================"
echo "Plotting AUC over wall-clock time"
echo "======================================================"

python src/analysis/plot_results_over_time_new.py \
    "$BASE_DIR/code/MedQuickTune/experiments/Baseline/$DATASET/$BASELINE_EXP" \
    "$BASE_DIR/code/MedQuickTune/experiments/NePS/$DATASET/$NEPS_EXP" \
    "$BASE_DIR/code/MedQuickTune/experiments/QuickTune/$DATASET/$QUICKTUNE_EXP" \
    --title "$TITLE" \
    --over-time \
    --cost-to-spend $COST_TO_SPEND \
    --output "$OUTPUT_DIR/${PLOT_NAME}.png"

echo "======================================================"
echo "Finished plotting"
echo "======================================================"