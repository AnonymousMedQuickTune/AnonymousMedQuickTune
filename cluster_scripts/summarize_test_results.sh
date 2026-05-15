#!/bin/bash
#SBATCH -p rome
#SBATCH -J summary
#SBATCH -t 01:59:59
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.err


source medquicktune/bin/activate

#EXPERIMENT_PATH=Baseline/lipo/densenet
#EXPERIMENT_PATH=NePS/lipo/BO
EXPERIMENT_PATH=QuickTune/lipo/MQ
SEED=42

BASE_DIR="projects/code/MedQuicktune"
EXPERIMENT_PATHS="$BASE_DIR/experiments/$EXPERIMENT_PATH"

cd "$BASE_DIR/code/MedQuickTune"
python -m src.analysis.summarize_evaluation_results "$EXPERIMENT_PATHS" --seed "$SEED" --output "$EXPERIMENT_PATHS/evaluation_summary_across_outer_fols.txt"
