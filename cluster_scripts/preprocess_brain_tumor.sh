#!/bin/bash
##SBATCH -p alldlc_gpu-rtx2080
#SBATCH -p testdlc_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
#SBATCH -J medquicktune
##SBATCH -t 23:59:59

source activate medquicktune

python -m src.classification_2d.preprocess_brain_tumor_data_2d \
    data.path=$DATA_PATH