#!/bin/bash
##SBATCH -p alldlc_gpu-rtx2080
#SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J medquicktune
##SBATCH -t 23:59:59

source activate medquicktune

BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"

python -m src.analysis.generalization_analysis \
    --dataset $DATASET \
    --exp1 $EXPERIMENT_NAME_1 \
    --exp2 $EXPERIMENT_NAME_2