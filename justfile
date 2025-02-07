# List all available recipes
list:
  just --list

# BASH SCRIPS --------------------------------------------------------------------------------------

# Format Python code using utils/format.sh
format:
  bash utils/format.sh

# Check code for errors and enforce style guidelines using Pylint
pylint:
  bash utils/pylint.sh

# Delete all experiments whose names start with 'test'
delete-tests:
  bash utils/delete_test_experiments.sh

# Download all the datasets
download-datasets:
  bash utils/download_datasets.sh

# Download a mini version of datasets for testing/debugging
download-mini-datasets:
  bash utils/download_mini_datasets.sh

# DATA PROCESSING ----------------------------------------------------------------------------------

# Process labels.csv into dataset-specific label files and create individual subject label files
preprocess-labels:
  python utils/preprocess_data.py

# Convert NePS output to QuickTune format (local machine)
neps2qt-local EXPERIMENT_NAME SEED:
  python src/analysis/neps_quicktune_output_adapter.py \
    experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/all_losses_and_configs.txt \
    --output-dir experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/quicktune_input

# Convert NePS output to QuickTune format (cluster)
neps2qt-cluster EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/neps2qt.sh

# Preprocess datasets locally for faster experiment initialization
preprocess-datasets:
    python -m src.preprocess_dataset

# Preprocess datasets on cluster for faster experiment initialization
preprocess-datasets-cluster:
    #!/usr/bin/env bash
    sbatch --exclude=dlcgpu05 \
        --output=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
        --error=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
        --export=DATA_PATH="/work/dlclarge1/wagnerd-medquicktune/datasets" \
        cluster_scripts/desmoid_preprocess_dataset.sh
# NEPS EXPERIMENTS ---------------------------------------------------------------------------------

# Run a test experiment on the local machine
run-local-test EXPERIMENT_NAME SEED:
  python -m src.train_neps \
    data.dataset=desmoid \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}}

# Submit a test experiment to the cluster
run-cluster-test EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/desmoid_train_neps.sh

# TEST EXPERIMENTS ---------------------------------------------------------------------------------

# Run test with best hyperparameters locally
test-local EXPERIMENT_NAME SEED:
  python -m src.test_best_config \
    --config_path experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/best_loss_with_config_trajectory.txt \
    --hydra_config configs/experiments/desmoid_config.yaml

# Run test with best hyperparameters on cluster
test-cluster EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},CONFIG_PATH="/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/best_loss_with_config_trajectory.txt",HYDRA_CONFIG="configs/experiments/desmoid_config.yaml" \
    cluster_scripts/desmoid_test_best_config.sh
