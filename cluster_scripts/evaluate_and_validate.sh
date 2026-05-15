#!/bin/bash
#SBATCH -J lipo-recover
#SBATCH -p gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=75G
#SBATCH -t 02:59:59
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out


source medquicktune/bin/activate

DATASET=lipo
SEED=42
EXPERIMENT_NAME=densenet
MODEL=densenet

BASE_DIR="/projects/code/MedQuickTune"

EXP_DIR="$BASE_DIR/experiments/Baseline/$DATASET/$EXPERIMENT_NAME/seed_$SEED"
NEPS_DIR="$EXP_DIR/NePS_output"

DATA_DIR="$BASE_DIR/data/worc"

PIPELINE_SPACE="configs/pipeline_spaces/baseline.yaml"

echo "============================================================"
echo "RECOVERING TEST EVALUATIONS"
echo "============================================================"

echo "Dataset: $DATASET"
echo "Experiment: $EXPERIMENT_NAME"
echo "Seed: $SEED"

python -m src.extras \
    run_mode=Baseline \
    model.type=$MODEL \
    model.task=classification \
    data.dataset=$DATASET \
    data.path=$DATA_DIR \
    experiment_base_dir=$EXP_DIR \
    neps_directory=$NEPS_DIR \
    pipeline_space=$PIPELINE_SPACE \
    seed=$SEED \
    data.dimensionality=3d \
    data.cache_data=True \
    data.use_smart_preprocessing=True \
    data.voxel_calculation=median \
    cv_outer_folds_repeats=1 \
    cv_outer_folds_splits=3 \
    cv_inner_folds_repeats=3 \
    cv_inner_folds_splits=3

echo "============================================================"
echo "FINISHED"
echo "============================================================"