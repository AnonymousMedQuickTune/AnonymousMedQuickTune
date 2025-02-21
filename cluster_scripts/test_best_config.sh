#!/bin/bash
##SBATCH -p alldlc_gpu-rtx2080
#SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J medquicktune
##SBATCH -t 23:59:59

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
EXP_DIR="$BASE_DIR/experiments/$DATASET/$EXPERIMENT_NAME/seed_$SEED"
DATA_DIR="$BASE_DIR/datasets/"

python -m src.test_best_config \
    --config_path "$CONFIG_PATH" \
    --hydra_config "$HYDRA_CONFIG" \
    --dataset "$DATASET" \
    --data_dir "$DATA_DIR" \
    --k_folds 5
