#!/bin/bash
#SBATCH -J lipoNEPS
#SBATCH -p gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 23:59:59
#SBATCH --array 0-2 # number of outer folds
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --propagate=NONE
#SBATCH --output=/projects/prjs1598/logs/%x-%j.out

source medquicktune/bin/activate

which python
python --version

DATASET=lipo
SEED=42
EXPERIMENT_NAME=bo

BASE_DIR="/projects/code/MedQuickTune"
EXP_DIR="$BASE_DIR/experiments/NePS/$DATASET/$EXPERIMENT_NAME/seed_$SEED"
DATA_DIR="$BASE_DIR/data/worc"

PIPELINE_SPACE="configs/pipeline_spaces/full.yaml"


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
    searcher=random_search
    #searcher=bayesian_optimization

echo "Finished"


