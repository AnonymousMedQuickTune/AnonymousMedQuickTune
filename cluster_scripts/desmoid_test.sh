#!/bin/bash
##SBATCH -p alldlc_gpu-rtx2080
#SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J medquicktune
##SBATCH -t 23:59:59
##SBATCH --array 0-5%3

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
EXP_DIR="$BASE_DIR/experiments/desmoid/$EXPERIMENT_NAME/seed_$SEED"
DATA_DIR="$BASE_DIR/datasets/"
PIPELINE_SPACE="configs/pipeline_spaces/pipeline_space_with_user_priors.yaml"

python -m src.train \
    data.dataset=desmoid \
    experiment_name=$EXPERIMENT_NAME \
    seed=$SEED \
    experiment_base_dir=$EXP_DIR \
    data.path=$DATA_DIR \
    pipeline_space=$PIPELINE_SPACE
