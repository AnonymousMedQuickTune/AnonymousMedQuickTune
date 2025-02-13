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
neps2qt-local DATASET EXPERIMENT_NAME SEED:
  python src/analysis/neps_quicktune_output_adapter.py \
    experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/all_losses_and_configs.txt \
    --output-dir experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/quicktune_input

# Convert NePS output to QuickTune format (cluster)
neps2qt-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/neps2qt.sh

# Preprocess datasets locally for faster experiment initialization
preprocess-datasets DATASET:
    python -m src.preprocess_dataset data.dataset={{DATASET}}

# Preprocess datasets on cluster for faster experiment initialization
preprocess-datasets-cluster DATASET:
    #!/usr/bin/env bash
    sbatch --exclude=dlcgpu05 \
        --output=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
        --error=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
        --export=DATA_PATH="/work/dlclarge1/wagnerd-medquicktune/datasets",DATASET={{DATASET}} \
        cluster_scripts/preprocess_dataset.sh

# NEPS EXPERIMENTS ---------------------------------------------------------------------------------

# Run a test experiment on the local machine with k-fold cross validation
run-local-test DATASET EXPERIMENT_NAME SEED:
  python -m src.train_neps \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \

# Submit a test experiment to the cluster
run-cluster-test DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/train_neps.sh

# TEST EXPERIMENTS ---------------------------------------------------------------------------------

# Run test with best hyperparameters locally
test-local DATASET EXPERIMENT_NAME SEED FOLDS:
  python -m src.test_best_config \
    --config_path experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/best_loss_with_config_trajectory.txt \
    --hydra_config configs/main_experiment_config.yaml \
    --dataset {{DATASET}} \
    --data_dir datasets \
    --k_folds {{FOLDS}}

# Run test with best hyperparameters on cluster
test-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},CONFIG_PATH="/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/best_loss_with_config_trajectory.txt",HYDRA_CONFIG="configs/main_experiment_config.yaml",DATA_DIR="/work/dlclarge1/wagnerd-medquicktune/datasets" \
    cluster_scripts/test_best_config.sh

# ANALYSIS -----------------------------------------------------------------------------------------

# Analyze and compare generalization performance between two NePS runs
analyze-generalization DATASET EXPERIMENT_NAME_1 EXPERIMENT_NAME_2:
  python -m src.analysis.generalization_analysis \
    --dataset {{DATASET}} \
    --exp1 {{EXPERIMENT_NAME_1}} \
    --exp2 {{EXPERIMENT_NAME_2}}

# Analyze and compare generalization performance between two NePS runs (cluster)
analyze-generalization-cluster DATASET EXPERIMENT_NAME_1 EXPERIMENT_NAME_2:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME_1={{EXPERIMENT_NAME_1}},EXPERIMENT_NAME_2={{EXPERIMENT_NAME_2}} \
    cluster_scripts/analyze_generalization.sh

# Analyze fidelity correlations from a NePS experiment
analyze-fidelity-correlation DATASET EXPERIMENT_NAME SEED:
    python -m src.analysis.fidelity_correlation \
        experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/all_losses_and_configs.txt

# Analyze fidelity correlations from a NePS experiment (cluster)
analyze-fidelity-correlation-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/analyze_fidelity_correlation.sh
