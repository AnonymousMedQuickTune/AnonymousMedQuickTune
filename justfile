# List all available recipes
list:
  just --list

# BASH SCRIPS --------------------------------------------------------------------------------------

# Format Python code using shell_scripts/format.sh
format:
  bash shell_scripts/format.sh

# Check code for errors and enforce style guidelines using Pylint
pylint:
  bash shell_scripts/pylint.sh

# Delete all experiments whose names start with 'test'
delete-tests:
  bash shell_scripts/delete_test_experiments.sh

# Download all the datasets
download-datasets:
  bash shell_scripts/download_datasets.sh

# Download a mini version of datasets for testing/debugging
download-mini-datasets:
  bash shell_scripts/download_mini_datasets.sh

# DATA PROCESSING ----------------------------------------------------------------------------------

# Convert NePS output to QuickTune format (local machine)
neps2qt-local DATASET EXP SEED:
  python src/analysis/neps_quicktune_output_adapter.py \
    experiments/{{DATASET}}/{{EXP}}/seed_{{SEED}} \
    --output-dir experiments/{{DATASET}}/{{EXP}}/seed_{{SEED}}/quicktune_input

# Convert NePS output to QuickTune format (cluster)
neps2qt-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/neps2qt.sh

# Preprocess brain tumor dataset: process raw data, create CSV and cache for faster experiment initialization
preprocess-brain-tumor-dataset:
    python -m src.classification_2d.preprocess_data_2d data.path=datasets

# Preprocess brain tumor dataset on cluster
preprocess-brain-tumor-cluster:
    #!/usr/bin/env bash
    sbatch --exclude=dlcgpu05 \
        --output=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
        --error=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
        --export=DATA_PATH="/work/dlclarge1/wagnerd-medquicktune/datasets" \
        cluster_scripts/preprocess_brain_tumor.sh

# NEPS EXPERIMENTS ---------------------------------------------------------------------------------

# Run an HPO experiment on the local machine
run-hpo-local DATASET EXPERIMENT_NAME SEED:
  python -m src.train_neps \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \

# Submit an HPO experiment to the cluster
run-hpo-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu19 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/train_neps.sh

# TEST EXPERIMENTS ---------------------------------------------------------------------------------

# Evaluate with best hyperparameter configuration
eval-local DATASET EXPERIMENT_NAME SEED FOLDS:
  python -m src.test_best_config \
    --config_path experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/best_loss_with_config_trajectory.txt \
    --hydra_config configs/main_experiment_config.yaml \
    --dataset {{DATASET}} \
    --data_dir datasets \
    --k_folds {{FOLDS}}

# Submit an evaluation of the best hyperparameter configuration to the cluster
eval-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},CONFIG_PATH="/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/best_loss_with_config_trajectory.txt",HYDRA_CONFIG="configs/main_experiment_config.yaml",DATA_DIR="/work/dlclarge1/wagnerd-medquicktune/datasets" \
    cluster_scripts/test_best_config.sh

# ANALYSIS -----------------------------------------------------------------------------------------

# Analyze and compare generalization performance between two NePS runs
analyze-generalization-local DATASET EXP1 SEED1 EXP2 SEED2:
  python -m src.analysis.generalization_analysis \
    --dataset {{DATASET}} \
    --exp1 {{EXP1}} \
    --seed1 {{SEED1}} \
    --exp2 {{EXP2}} \
    --seed2 {{SEED2}}

# Analyze and compare generalization performance between two NePS runs (cluster)
analyze-generalization-cluster DATASET EXP1 SEED1 EXP2 SEED2:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME_1={{EXP1}},SEED1={{SEED1}},EXPERIMENT_NAME_2={{EXP2}},SEED2={{SEED2}} \
    cluster_scripts/analyze_generalization.sh

# Analyze fidelity correlations from a NePS experiment
analyze-fidelity-correlation-local DATASET EXPERIMENT_NAME SEED:
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

# Merge multiple NePS runs into a single QuickTune portfolio
merge-neps2qt-local DATASET EXPERIMENT_NAMES SEEDS OUTPUT_DIR:
  python src/analysis/neps_quicktune_output_adapter.py \
    experiments/{{DATASET}} \
    --merge-runs \
    --experiment-names "{{EXPERIMENT_NAMES}}" \
    --seeds "{{SEEDS}}" \
    --output-dir {{OUTPUT_DIR}}

# Run QuickTune on a portfolio of NePS runs
run-quicktune-local DATASET EXPERIMENT_NAME SEED PORTFOLIO_DIR USE_MEDICAL_PORTFOLIO="true":
  python -m src.run_quicktune \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \
    portfolio_dir={{PORTFOLIO_DIR}} \
    data.path=datasets \
    use_medical_portfolio={{USE_MEDICAL_PORTFOLIO}}

# Evaluate NePS optimization results
eval-neps-local DATASET EXPERIMENT_NAME SEED:
  python -m src.evaluate_neps \
    experiment_name={{EXPERIMENT_NAME}} \
    data.dataset={{DATASET}} \
    seed={{SEED}} \
    data.path=datasets

# Submit a NePS evaluation to the cluster
eval-neps-cluster DATASET EXPERIMENT_NAME SEED FOLDS:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},FOLDS={{FOLDS}},EXPERIMENT_DIR="/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}",HYDRA_CONFIG="configs/main_experiment_config.yaml",DATA_DIR="/work/dlclarge1/wagnerd-medquicktune/datasets" \
    cluster_scripts/evaluate_neps.sh
