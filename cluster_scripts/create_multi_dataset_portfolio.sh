#!/bin/bash

# ======================================================
# Portfolio creation script for MedQuickTune.
#
# This script generates the portfolio used for the
# portfolio-based meta-learning experiments presented
# in the paper.
#
# The portfolio is constructed by aggregating previous
# NePS optimization runs across multiple medical imaging
# datasets and converting them into a unified format
# compatible with MedQuickTune.
#
# The generated portfolio contains:
#   - Hyperparameter configurations
#   - Learning curves
#   - Runtime costs
#   - Dataset meta-features
#
# These optimization trajectories are later used by
# MedQuickTune to initialize and guide hyperparameter
# optimization on unseen target datasets.
#
# Portfolio construction uses Random Search
# (searcher=random_search) to encourage broader and
# less biased exploration of the search space, improving
# the diversity and robustness of the collected
# optimization trajectories for meta-learning.
# ======================================================

#SBATCH -p gpu_a100
#SBATCH --gres=gpu:1
#SBATCH -J create_portfolio

##SBATCH -t 23:59:59

#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out
#SBATCH --output=/projects/code/MedQuickTune/logs/%x-%j.out

# Activate MedQuickTune environment
source medquicktune/bin/activate

which python
python --version

# Base project directory
BASE_DIR="/projects/code/MedQuickTune"

# Output portfolio directory
PORTFOLIO_DIR="$BASE_DIR/experiments/Portfolio"

# Example portfolio specification for Lipo Portfolio:
# gist:full_portfolio(42);
# desmoid:full_portfolio(42);
# hcc:full_portfolio(42);
# bflair:full_portfolio(42);
# liver:full_portfolio(42);
# crlm:full_portfolio(42);

python -m src.analysis.create_portfolio \
    +dataset_spec="'{{DATASET_SPEC}}'" \
    portfolio_dir=$PORTFOLIO_DIR \
    merge_runs=true \
    +multi_dataset=true \
    run_mode=Portfolio \
    hydra.run.dir=$BASE_DIR/experiments/Portfolio/logs

echo "======================================================"
echo "Finished portfolio creation"
echo "Portfolio successfully generated"
echo "======================================================"