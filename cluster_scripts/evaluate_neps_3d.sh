#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
##SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J medquicktune-eval
##SBATCH -t 23:59:59

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
EXP_DIR="$BASE_DIR/experiments/NePS/$DATASET/$EXPERIMENT_NAME/seed_$SEED"
DATA_DIR="$BASE_DIR/datasets/"

python -m src.evaluate_neps \
    experiment_name=$EXPERIMENT_NAME \
    data.dataset=$DATASET \
    seed=$SEED \
    data.path=$DATA_DIR \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=false
