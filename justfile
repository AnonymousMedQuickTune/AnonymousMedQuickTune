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

# Plot CV results from MedQuickTune experiment directories
plot-cv-results EXPERIMENT_PATH OUTPUT_DIR="results/lipo_baseline":
  python src/analysis/plot_cluster_results.py {{EXPERIMENT_PATH}} --auto-structure --output-dir {{OUTPUT_DIR}}

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
run-2d-hpo-local DATASET EXPERIMENT_NAME SEED:
  python -m src.train_neps \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \
    model.type=resnet \
    data.dimensionality=2d \
    developer_mode=true

# Run an HPO experiment on a 3D dataset on the local machine
run-3d-hpo-local DATASET EXPERIMENT_NAME SEED:
  python -m src.train_neps \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true

# Run multiple HPO experiments sequentially for portfolio cration on the local machine
run-portfolio-experiments:
  # Lipo
  python -m src.train_neps \
    data.dataset=lipo \
    experiment_name=test_portfolio_1 \
    seed=42 \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.evaluate_neps \
    experiment_name=test_portfolio_1 \
    data.dataset=lipo \
    seed=42 \
    data.path=datasets \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.train_neps \
    data.dataset=lipo \
    experiment_name=test_portfolio_1 \
    seed=43 \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.evaluate_neps \
    experiment_name=test_portfolio_1 \
    data.dataset=lipo \
    seed=43 \
    data.path=datasets \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.train_neps \
    data.dataset=lipo \
    experiment_name=test_portfolio_2 \
    seed=43 \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.evaluate_neps \
    experiment_name=test_portfolio_2 \
    data.dataset=lipo \
    seed=43 \
    data.path=datasets \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.train_neps \
    data.dataset=lipo \
    experiment_name=test_portfolio_2 \
    seed=44 \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.evaluate_neps \
    experiment_name=test_portfolio_2 \
    data.dataset=lipo \
    seed=44 \
    data.path=datasets \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  # Desmoid
  python -m src.train_neps \
    data.dataset=desmoid \
    experiment_name=test_portfolio_1 \
    seed=42 \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true

  python -m src.evaluate_neps \
    experiment_name=test_portfolio_1 \
    data.dataset=desmoid \
    seed=42 \
    data.path=datasets \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true

  python -m src.train_neps \
    data.dataset=desmoid \
    experiment_name=test_portfolio_1 \
    seed=43 \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.evaluate_neps \
    experiment_name=test_portfolio_1 \
    data.dataset=desmoid \
    seed=43 \
    data.path=datasets \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.train_neps \
    data.dataset=desmoid \
    experiment_name=test_portfolio_2 \
    seed=43 \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.evaluate_neps \
    experiment_name=test_portfolio_2 \
    data.dataset=desmoid \
    seed=43 \
    data.path=datasets \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.train_neps \
    data.dataset=desmoid \
    experiment_name=test_portfolio_2 \
    seed=44 \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.evaluate_neps \
    experiment_name=test_portfolio_2 \
    data.dataset=desmoid \
    seed=44 \
    data.path=datasets \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true

# Submit an HPO experiment for a 2D dataset to the cluster
run-2d-neps-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu19 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/train_neps_2d.sh

# Submit an HPO experiment for a 3D dataset to the cluster
run-3d-neps-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu19 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/train_neps_3d.sh

# TEST EXPERIMENTS ---------------------------------------------------------------------------------

# Evaluate with best hyperparameter configuration
eval-local DATASET EXPERIMENT_NAME SEED FOLDS:
  python -m src.test_best_config \
    --config_path experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/best_loss_with_config_trajectory.txt \
    --hydra_config configs/experimental_setting.yaml \
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
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},CONFIG_PATH="/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/NePS_output/best_loss_with_config_trajectory.txt",HYDRA_CONFIG="configs/experimental_setting.yaml",DATA_DIR="/work/dlclarge1/wagnerd-medquicktune/datasets" \
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
create-portfolio-local DATASET EXPERIMENT_NAMES SEEDS:
  python -m src.analysis.neps_quicktune_output_adapter \
    data.dataset="{{DATASET}}" \
    experiment_names="'{{EXPERIMENT_NAMES}}'" \
    seeds="'{{SEEDS}}'" \
    portfolio_dir=experiments/Portfolio \
    merge_runs=true \
    run_mode=Portfolio \
    experiment_dir_suffix=""

# Run QuickTune on a portfolio of NePS runs
run-quicktune-local DATASET EXPERIMENT_NAME SEED PORTFOLIO_DIR USE_MEDICAL_PORTFOLIO="true":
  python -m src.run_quicktune \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \
    portfolio_dir={{PORTFOLIO_DIR}} \
    data.path=datasets \
    run_mode=QuickTune \
    qt.use_medical_portfolio={{USE_MEDICAL_PORTFOLIO}}

# Evaluate NePS optimization results for a 2D dataset
eval-2d-neps-local DATASET EXPERIMENT_NAME SEED:
  python -m src.evaluate_neps \
    experiment_name={{EXPERIMENT_NAME}} \
    data.dataset={{DATASET}} \
    seed={{SEED}} \
    data.path=datasets

# Evaluate NePS optimization results for a 3D dataset
eval-3d-neps-local DATASET EXPERIMENT_NAME SEED:
  python -m src.evaluate_neps \
    experiment_name={{EXPERIMENT_NAME}} \
    data.dataset={{DATASET}} \
    seed={{SEED}} \
    data.path=datasets \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true

# Submit a NePS evaluation to the cluster for a 2D dataset
eval-2d-neps-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},EXPERIMENT_DIR="/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}",HYDRA_CONFIG="configs/experimental_setting.yaml",DATA_DIR="/work/dlclarge1/wagnerd-medquicktune/datasets" \
    cluster_scripts/evaluate_neps_2d.sh

# Submit a NePS evaluation to the cluster for a 3D dataset
eval-3d-neps-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}},EXPERIMENT_DIR="/work/dlclarge1/wagnerd-medquicktune/experiments/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}",HYDRA_CONFIG="configs/experimental_setting.yaml",DATA_DIR="/work/dlclarge1/wagnerd-medquicktune/datasets" \
    cluster_scripts/evaluate_neps_3d.sh
