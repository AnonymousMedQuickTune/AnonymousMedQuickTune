#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gres=gpu:1
#SBATCH -J create_portfolio
##SBATCH -t 23:59:59
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out


source medquicktune/bin/activate

which python
python --version

BASE_DIR="/projects/code/MedQuickTune"
PORTFOLIO_DIR="$BASE_DIR/experiments/Portfolio"

# Example portfolio: lipo:full_portfolio(42);desmoid:full_portfolio(42);hcc:full_portfolio(42);bflair:full_portfolio(42);liver:full_portfolio(42);crlm:full_portfolio(42);

python -m src.analysis.create_portfolio \
    +dataset_spec="'{{DATASET_SPEC}}'" \
    portfolio_dir=$PORTFOLIO_DIR \
    merge_runs=true \
    +multi_dataset=true \
    run_mode=Portfolio \
    hydra.run.dir=$BASE_DIR/experiments/Portfolio/logs

#  