#!/bin/bash
##SBATCH -p alldlc_gpu-rtx2080
#SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J medquicktune
##SBATCH -t 23:59:59

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
EXPERIMENT_PATH="$BASE_DIR/experiments/$EXPERIMENT_PATH"

cd "$BASE_DIR/code/MedQuickTune"
python -m src.analysis.summarize_evaluation_results "$EXPERIMENT_PATH" --seed "$SEED" --output "$EXPERIMENT_PATH/evaluation_summary_across_outer_fols.txt"
