#!/bin/bash
##SBATCH -p alldlc_gpu-rtx2080
#SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J medquicktune
##SBATCH -t 23:59:59

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
INPUT_FILE="$BASE_DIR/experiments/$DATASET/$EXPERIMENT_NAME/seed_$SEED/NePS_output/all_losses_and_configs.txt"

python -m src.analysis.fidelity_correlation \
    "$INPUT_FILE"