#!/bin/bash

# ======================================================
# Script for generating the AUC vs wall-clock time plots
# presented in the Results section of the paper.
#
# This script compares:
#   1. Baseline
#   2. Standard Bayesian Optimization (BO)
#   3. MedQuickTune
#
# The generated plots visualize performance over time
# under a fixed computational budget.
# ======================================================

##SBATCH -p gpu_a100
#SBATCH -p rome
##SBATCH --gres=gpu:1
#SBATCH -J plot_auc_time
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out
#SBATCH --error=/projects/code/MedQuickTune/logs/%x-%j.err

# Activate MedQuickTune environment
source /medquicktune/bin/activate

# Base project directory
BASE_DIR="projects/code/MedQuickTune"

# Dataset used for plotting
DATASET="lipo"

# Experiment names
BASELINE_EXP="densenet"
NEPS_EXP="bo"
QUICKTUNE_EXP="medquicktune"

# Output plot configuration
PLOT_NAME="${DATASET}_auc_comparison"
TITLE="WORC Lipo Dataset"

# Total compute budget in seconds (24 hours)
COST_TO_SPEND=86400

# Directory where plots will be saved
OUTPUT_DIR="$BASE_DIR/experiments/Plots"
mkdir -p "$OUTPUT_DIR"

# Move to repository root
cd "$BASE_DIR/code/MedQuickTune"

echo "======================================================"
echo "Plotting AUC over wall-clock time"
echo "Generating figures for the Results section"
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
echo "Figure successfully generated"
echo "======================================================"