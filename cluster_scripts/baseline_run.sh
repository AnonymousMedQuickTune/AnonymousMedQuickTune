#!/bin/bash
#SBATCH -J lipo-densenet
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH --mem=75G
#SBATCH -t 10:59:99
##SBATCH --array=0-2
#SBATCH --output=/projects/prjs1598/logs/%x-%j.out

source medquicktune/bin/activate

DATASET=lipo
SEED=42
EXPERIMENT_NAME=densenet
MODEL=densenet

BASE_DIR="/projects/prjs1598"
EXP_DIR="$BASE_DIR/experiments/Baseline/$DATASET/$EXPERIMENT_NAME/seed_$SEED"
DATA_DIR="$BASE_DIR/data/worc"
#PIPELINE_SPACE="configs/pipeline_spaces/pipeline_space_without_user_priors.yaml"
PIPELINE_SPACE="configs/pipeline_spaces/baseline.yaml"


echo "Starting"


python -m src.run_neps \
    data.dataset=$DATASET \
    experiment_name=$EXPERIMENT_NAME \
    seed=$SEED \
    experiment_base_dir=$EXP_DIR \
    data.path=$DATA_DIR \
    pipeline_space=$PIPELINE_SPACE \
    model.type=$MODEL \
    data.dimensionality=3d \
    run_mode="Baseline" \
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
    data.num_workers=4

echo "Finished"


