#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
##SBATCH -p mlhiwidlc_gpu-rtx2080
##SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J create_portfolio
##SBATCH -t 23:59:59

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
PORTFOLIO_DIR="$BASE_DIR/experiments/Portfolio"

python -m src.analysis.create_portfolio \
    +dataset_spec="'{{DATASET_SPEC}}'" \
    portfolio_dir=$PORTFOLIO_DIR \
    merge_runs=true \
    +multi_dataset=true \
    run_mode=Portfolio \
    hydra.run.dir=$BASE_DIR/experiments/Portfolio/logs
