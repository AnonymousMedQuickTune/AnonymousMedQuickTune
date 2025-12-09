#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
##SBATCH -p mldlc2_gpu-l40s
##SBATCH -p mlhiwidlc_gpu-rtx2080
##SBATCH -p testdlc_gpu-rtx2080
##SBATCH --mem-per-gpu=32G
##SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --propagate=NONE
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J medquicktune
##SBATCH -t 23:59:59
#SBATCH --array 1-3%3  # number of outer folds

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
EXP_DIR="$BASE_DIR/experiments/NePS/$DATASET/$EXPERIMENT_NAME/seed_$SEED"
DATA_DIR="$BASE_DIR/datasets/"
# PIPELINE_SPACE="configs/pipeline_spaces/pipeline_space_without_user_priors.yaml"
PIPELINE_SPACE="configs/pipeline_spaces/densenet.yaml"

python -m src.run_quicktune \
    data.dataset=$DATASET \
    experiment_name=$EXPERIMENT_NAME \
    seed=$SEED \
    portfolio_dir=$PORTFOLIO_DIR \
    data.path=$DATA_DIR \
    data.dimensionality=3d \
    run_mode=QuickTune \
    qt.use_medical_portfolio=$USE_MEDICAL_PORTFOLIO \
    pipeline_space=$PIPELINE_SPACE \
    combine_model_and_training_space=False \
    training.number_of_epochs=25 \
    cv_inner_folds_repeats=3 \
    cv_inner_folds_splits=3 \
    cv_outer_folds_repeats=1 \
    cv_outer_folds_splits=3 \
    developer_mode=false \
    data.augmentation_type="basic" \
    training.early_stopping=True \
    training.patience=10 \
    training.scheduler_type=warmup \
    validation_evaluation=ensemble \
    data.num_workers=4 \
    cost_to_spend=85800  # 24 hours - 10 min
