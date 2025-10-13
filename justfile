# List all available recipes
list:
  just --list

# --------------------------------------------------------------------------------------------------
# BASH SCRIPS
# --------------------------------------------------------------------------------------------------

# Format Python code using shell_scripts/format.sh
format:
  bash shell_scripts/format.sh

# Check code for errors and enforce style guidelines using Pylint
pylint:
  bash shell_scripts/pylint.sh

# Delete all experiments whose names start with 'test'
delete-tests:  # TODO @Diane: test if this still works!
  bash shell_scripts/delete_test_experiments.sh

# Download all the datasets
download-datasets:  # TODO @Diane: implement or delete this!
  bash shell_scripts/download_datasets.sh

# Download a mini version of datasets for testing/debugging
download-mini-datasets:  # TODO @Diane: implement or delete this!
  bash shell_scripts/download_mini_datasets.sh

# --------------------------------------------------------------------------------------------------
# DATA PROCESSING
# --------------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------
# NEPS EXPERIMENTS
# --------------------------------------------------------------------------------------------------

# Run an HPO experiment with NePSon the local machine for a 2D dataset
run-2d-neps-local DATASET EXPERIMENT_NAME SEED:
  python -m src.run_neps \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \
    model.type=resnet \
    data.dimensionality=2d \
    developer_mode=true

# Run an HPO experiment with NePS on the local machine for a 3D dataset
run-3d-neps-local DATASET MODEL EXPERIMENT_NAME:
  python -m src.run_neps \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed=42 \
    model.type={{MODEL}} \
    data.dimensionality=3d \
    developer_mode=true

# Run a baseline HPO experiment with fixed hyperparames on the local machine for a 3D dataset
run-3d-baseline-local DATASET MODEL EXPERIMENT_NAME:
  python -m src.run_neps \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed=42 \
    model.type={{MODEL}} \
    data.dimensionality=3d \
    developer_mode=true \
    run_mode="Baseline"

# Run multiple HPO experiments sequentially for portfolio cration with NePS on the local machine for 3D datasets
run-portfolio-test-experiments:
  # Lipo
  python -m src.run_neps \
    data.dataset=lipo \
    experiment_name=test_portfolio_1 \
    seed=42 \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.run_neps \
    data.dataset=lipo \
    experiment_name=test_portfolio_1 \
    seed=43 \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.run_neps \
    data.dataset=lipo \
    experiment_name=test_portfolio_2 \
    seed=43 \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.run_neps \
    data.dataset=lipo \
    experiment_name=test_portfolio_2 \
    seed=44 \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  # Desmoid
  python -m src.run_neps \
    data.dataset=desmoid \
    experiment_name=test_portfolio_1 \
    seed=42 \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true

  python -m src.run_neps \
    data.dataset=desmoid \
    experiment_name=test_portfolio_1 \
    seed=43 \
    model.type=densenetv1 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.run_neps \
    data.dataset=desmoid \
    experiment_name=test_portfolio_2 \
    seed=43 \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true
  
  python -m src.run_neps \
    data.dataset=desmoid \
    experiment_name=test_portfolio_2 \
    seed=44 \
    model.type=densenetv2 \
    data.dimensionality=3d \
    developer_mode=true

# Submit an HPO experiment with NePS to the cluster for a 2D dataset      
run-2d-neps-cluster DATASET EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu19 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} \
    cluster_scripts/run_neps_2d.sh

# Submit an HPO experiment with NePS to the cluster for a 3D dataset
run-3d-neps-cluster DATASET MODEL EXPERIMENT_NAME:
  #!/usr/bin/env bash
  SEED=42
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/
  sbatch --exclude=dlcgpu19 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/NePS/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED=${SEED},MODEL={{MODEL}} \
    cluster_scripts/run_neps_3d.sh

# Submit a baseline experiment with fixed hyperparameters to the cluster for a 3D dataset
run-3d-baseline-cluster DATASET MODEL EXPERIMENT_NAME:
  #!/usr/bin/env bash
  SEED=42
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/Baseline/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/
  sbatch --exclude=dlcgpu19 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/Baseline/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/Baseline/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED=${SEED},MODEL={{MODEL}} \
    cluster_scripts/run_baseline_3d.sh

# --------------------------------------------------------------------------------------------------
# PORTFOLIO EXPERIMENTS
# --------------------------------------------------------------------------------------------------

# example: just create-multi-dataset-portfolio "lipo:test_portfolio_1(42,43),test_portfolio_2(43,44);desmoid:test_portfolio_1(42,43),test_portfolio_2(43,44)"
# Merge multiple NePS runs from multiple datasets into a single QuickTune portfolio
create-multi-dataset-portfolio DATASET_SPEC:
  python -m src.analysis.create_portfolio \
    +dataset_spec="'{{DATASET_SPEC}}'" \
    portfolio_dir=experiments/Portfolio \
    merge_runs=true \
    +multi_dataset=true \
    run_mode=Portfolio \
    hydra.run.dir=experiments/Portfolio/logs

# --------------------------------------------------------------------------------------------------
# QUICKTUNE EXPERIMENTS
# --------------------------------------------------------------------------------------------------

# Run an HPO experiment with QuickTune for 2d datasets on a portfolio of NePS runs
run-2d-quicktune-local DATASET EXPERIMENT_NAME SEED PORTFOLIO_DIR USE_MEDICAL_PORTFOLIO="true":
  python -m src.run_quicktune \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \
    portfolio_dir={{PORTFOLIO_DIR}} \
    data.path=datasets \
    run_mode=QuickTune \
    qt.use_medical_portfolio={{USE_MEDICAL_PORTFOLIO}} \
    developer_mode=true
  
# Run an HPO experiment with QuickTune for 3d datasets on a portfolio of NePS runs
run-3d-quicktune-local DATASET EXPERIMENT_NAME SEED PORTFOLIO_DIR USE_MEDICAL_PORTFOLIO="true":
  python -m src.run_quicktune \
    data.dataset={{DATASET}} \
    experiment_name={{EXPERIMENT_NAME}} \
    seed={{SEED}} \
    portfolio_dir={{PORTFOLIO_DIR}} \
    data.path=datasets \
    data.dimensionality=3d \
    run_mode=QuickTune \
    qt.use_medical_portfolio={{USE_MEDICAL_PORTFOLIO}} \
    developer_mode=true

# --------------------------------------------------------------------------------------------------
# ANALYSIS  # TODO @Diane: test if this still works or delete this!
# --------------------------------------------------------------------------------------------------

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
