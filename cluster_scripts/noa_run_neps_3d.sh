#!/bin/bash
#SBATCH -J medquicktune
#SBATCH -n 1
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G                   # adjust if needed
#SBATCH -t 23:59:00
#SBATCH --output=/projects/prjs1598/logs/%x-%j.out


# --- Modules (your style) ---
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# module load CUDA/11.7.0           # only if your env needs a specific CUDA toolkit


source medquicktune/bin/activate


BASE_DIR="/projects/prjs1598"
EXP_DIR="$BASE_DIR/experiments/NePS/$DATASET/$EXPERIMENT_NAME/seed_$SEED"
DATA_DIR="$BASE_DIR/data/worc/"
#PIPELINE_SPACE="configs/pipeline_spaces/pipeline_space_without_user_priors.yaml"
PIPELINE_SPACE="configs/pipeline_spaces/baseline.yaml"


echo "Starting"


python -m src.train_neps \
   data.dataset=$DATASET \
   experiment_name=$EXPERIMENT_NAME \
   seed=$SEED \
   experiment_base_dir=$EXP_DIR \
   data.path=$DATA_DIR \
   pipeline_space=$PIPELINE_SPACE \
   model.type=densenetv2 \
   data.dimensionality=3d \
   # developer_mode=true


echo "Finished"


