#!/bin/bash

# ======================================================
# NePS hyperparameter optimization experiment script
# for 3D medical image classification on the WORC Lipo
# dataset.
#
# This script reproduces the NePS optimization results
# presented in the Results section of the paper.
#
# The experiment performs:
#   - Compute-bounded hyperparameter optimization
#   - 3D model selection and training
#   - Nested cross-validation evaluation
#
# The generated outputs are later used to create the
# AUC vs wall-clock time comparison figures between:
#   1. Baseline
#   2. Bayesian Optimization (BO)
#   3. MedQuickTune
#
# Search strategy:
#   - Bayesian Optimization experiments are performed
#     using:
#         searcher=bayesian_optimization
#
#   - Portfolio construction experiments are performed
#     using:
#         searcher=random_search
#
# Random Search is used during portfolio generation to
# encourage broader and less biased exploration of the
# search space, which improves the diversity of the
# collected optimization trajectories used for
# meta-learning.
# ======================================================

#SBATCH -J lipoNEPS
#SBATCH -p gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 23:59:59
#SBATCH --array 0-2 # Number of outer folds
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --propagate=NONE
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out

# Activate MedQuickTune environment
source medquicktune/bin/activate

which python
python --version

# Dataset configuration
DATASET=lipo
SEED=42
EXPERIMENT_NAME=bo

# Base project directory
BASE_DIR="/projects/code/MedQuickTune"

# Output experiment directory
EXP_DIR="$BASE_DIR/experiments/NePS/$DATASET/$EXPERIMENT_NAME/seed_$SEED"

# Dataset directory
DATA_DIR="$BASE_DIR/data/worc"

# Hyperparameter search space configuration
PIPELINE_SPACE="configs/pipeline_spaces/full.yaml"

echo "======================================================"
echo "Starting NePS experiment"
echo "Dataset: $DATASET"
echo "Generating results for the paper"
echo "======================================================"

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
    #searcher=random_search
    searcher=bayesian_optimization

echo "======================================================"
echo "Finished NePS experiment"
echo "Results successfully generated"
echo "======================================================"