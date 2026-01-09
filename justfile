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

# Analyze most expensive dataset-model-voxel_calculation combinations for 50 epochs
# Example: just analyze-expensive-configs (uses values from experimental_setting.yaml)
# Example with custom CV settings: just analyze-expensive-configs 5 3 5
# Example with custom top N: just analyze-expensive-configs 5 3 5 100
analyze-expensive-configs CV_OUTER_FOLDS_REPEATS CV_OUTER_FOLDS_SPLITS CV_INNER_FOLDS TOP_N:
  #!/usr/bin/env bash
  ARGS=""
  if [ -n "{{CV_OUTER_FOLDS_REPEATS}}" ]; then
    ARGS="${ARGS} --cv-outer-folds-repeats {{CV_OUTER_FOLDS_REPEATS}}"
  fi
  if [ -n "{{CV_OUTER_FOLDS_SPLITS}}" ]; then
    ARGS="${ARGS} --cv-outer-folds-splits {{CV_OUTER_FOLDS_SPLITS}}"
  fi
  if [ -n "{{CV_INNER_FOLDS}}" ]; then
    ARGS="${ARGS} --cv-inner-folds {{CV_INNER_FOLDS}}"
  fi
  if [ -n "{{TOP_N}}" ]; then
    ARGS="${ARGS} --top-n {{TOP_N}}"
  fi
  python src/analysis/analyze_most_expensive_configs.py ${ARGS}

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
    developer_mode=true \
    cost_to_spend=60

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

# example: just create-multi-dataset-portfolio "lipo:test_portfolio_1(42,43),test_portfolio_2(43,44);desmoid:test_portfolio_1(42,43),test_portfolio_2(43,44)" portfolio_name
# Merge multiple NePS runs from multiple datasets into a single QuickTune portfolio
create-multi-dataset-portfolio DATASET_SPEC PORTFOLIO_NAME:
  python -m src.analysis.create_portfolio \
    +dataset_spec="'{{DATASET_SPEC}}'" \
    +portfolio_name="{{PORTFOLIO_NAME}}" \
    portfolio_dir=experiments/Portfolio \
    merge_runs=true \
    +multi_dataset=true \
    run_mode=Portfolio \
    hydra.run.dir=experiments/Portfolio/{{PORTFOLIO_NAME}}/logs

# Merge multiple NePS runs from multiple datasets into a single QuickTune portfolio on the cluster
create-multi-dataset-portfolio-cluster DATASET_SPEC PORTFOLIO_NAME:
  python -m src.analysis.create_portfolio_cluster \
    +dataset_spec="'{{DATASET_SPEC}}'" \
    +portfolio_name="{{PORTFOLIO_NAME}}" \
    portfolio_dir=/work/dlclarge1/wagnerd-medquicktune/experiments/Portfolio \
    merge_runs=true \
    +multi_dataset=true \
    run_mode=Portfolio \
    hydra.run.dir=/work/dlclarge1/wagnerd-medquicktune/experiments/Portfolio/{{PORTFOLIO_NAME}}/logs

# example: just create-multi-dataset-portfolio-cluster "lipo:test_portfolio_1(42,43),test_portfolio_2(43,44);desmoid:test_portfolio_1(42,43),test_portfolio_2(43,44)"
# Merge multiple NePS runs from multiple datasets into a single QuickTune portfolio (cluster)
create-multi-dataset-portfolio-cluster-submission DATASET_SPEC:
  #!/usr/bin/env bash
  BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
  mkdir -p ${BASE_DIR}/experiments/Portfolio/cluster_oe/
  # Escape the DATASET_SPEC to prevent shell interpretation of special characters like parentheses
  ESCAPED_SPEC=$(printf '%q' "{{DATASET_SPEC}}")
  sbatch --exclude=dlcgpu19 \
    --output=${BASE_DIR}/experiments/Portfolio/cluster_oe/%x.%A.%a.%N.err_out \
    --error=${BASE_DIR}/experiments/Portfolio/cluster_oe/%x.%A.%a.%N.err_out \
    --export="DATASET_SPEC={{DATASET_SPEC}}" \
    cluster_scripts/create_multi_dataset_portfolio.sh
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
    developer_mode=true \
    cost_to_spend=600

# Submit an HPO experiment with QuickTune to the cluster for a 3D dataset
run-3d-quicktune-cluster DATASET EXPERIMENT_NAME PORTFOLIO_DIR USE_MEDICAL_PORTFOLIO="true":
  #!/usr/bin/env bash
  SEED=42
  PORTFOLIO_DIR=/work/dlclarge1/wagnerd-medquicktune/code/MedQuickTune/experiments/Portfolio
  USE_MEDICAL_PORTFOLIO="true"
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/QuickTune/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/
  sbatch --exclude=dlcgpu19 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/QuickTune/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/QuickTune/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=DATASET={{DATASET}},EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED=${SEED},PORTFOLIO_DIR={{PORTFOLIO_DIR}},USE_MEDICAL_PORTFOLIO={{USE_MEDICAL_PORTFOLIO}} \
    cluster_scripts/run_quicktune_3d.sh

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

# Summarize test results across all cross-validation folds for NePS experiments
summarize-test-results EXPERIMENT_PATH SEED="42":
  python -m src.analysis.summarize_evaluation_results {{EXPERIMENT_PATH}} --seed {{SEED}}

# Summarize test results across all cross-validation folds for NePS experiments (cluster)
summarize-test-results-cluster EXPERIMENT_PATH SEED="42":
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/cluster_oe/
  sbatch --exclude=dlcgpu05 \
    --output=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/cluster_oe/%x.%A.%a.%N.err_out \
    --export=EXPERIMENT_PATH={{EXPERIMENT_PATH}},SEED={{SEED}} \
    cluster_scripts/summarize_test_results.sh

# Plot test and validation performance over time (number of configs) for NePS experiments (single experiment)
# Example: just plot-performance-over-time experiments/NePS/lipo/test_plotting_script
# Example with custom output: just plot-performance-over-time experiments/NePS/lipo/test_plotting_script output.png
plot-performance-over-time EXPERIMENT_DIR OUTPUT_PATH="":
  #!/usr/bin/env bash
  if [ -z "{{OUTPUT_PATH}}" ]; then
    python src/analysis/plot_results_over_time.py {{EXPERIMENT_DIR}}
  else
    python src/analysis/plot_results_over_time.py {{EXPERIMENT_DIR}} --output {{OUTPUT_PATH}}
  fi

# Update cost CSV files for NePS experiments (creates costs_in_sec.csv, costs_in_min.csv, costs_in_hours.csv)
# Example: just update-neps-cost-csv experiments/NePS/lipo/test_full_final
# Example with seed: just update-neps-cost-csv experiments/NePS/lipo/test_full_final/seed_42
update-neps-cost-csv EXPERIMENT_PATH:
  #!/usr/bin/env bash
  # If path ends with seed_*, extract NePS_output directory
  if [[ "{{EXPERIMENT_PATH}}" == */seed_* ]]; then
    NEPS_OUTPUT_DIR="{{EXPERIMENT_PATH}}/NePS_output"
  else
    # Find first seed directory
    SEED_DIRS=$(find "{{EXPERIMENT_PATH}}" -maxdepth 1 -type d -name "seed_*" | head -1)
    if [ -z "$SEED_DIRS" ]; then
      echo "Error: No seed directories found in {{EXPERIMENT_PATH}}"
      exit 1
    fi
    NEPS_OUTPUT_DIR="${SEED_DIRS}/NePS_output"
  fi
  
  if [ ! -d "$NEPS_OUTPUT_DIR" ]; then
    echo "Error: NePS_output directory not found: $NEPS_OUTPUT_DIR"
    exit 1
  fi
  
  python -c "from src.utils.logging_utils import update_cost_csv_from_neps_output; update_cost_csv_from_neps_output('$NEPS_OUTPUT_DIR')"

# Plot test and validation performance over time (wall-clock time) for NePS experiments
# Requires cost_to_spend parameter (total time budget in seconds)
# Automatically creates cost CSV files if they don't exist
# Example: just plot-neps-over-time experiments/NePS/lipo/test_full_final 86400
# Example with seed: just plot-neps-over-time experiments/NePS/lipo/test_full_final/seed_42 86400
# Example with custom output: just plot-neps-over-time experiments/NePS/lipo/test_full_final 86400 output.png
# Note: cost_to_spend is typically 86400 (24 hours) or 1800 (30 minutes) for developer mode
plot-neps-over-time EXPERIMENT_PATH COST_TO_SPEND OUTPUT_PATH="":
  #!/usr/bin/env bash
  # If path ends with seed_*, go up one level to get experiment directory
  if [[ "{{EXPERIMENT_PATH}}" == */seed_* ]]; then
    EXPERIMENT_DIR=$(dirname "{{EXPERIMENT_PATH}}")
  else
    EXPERIMENT_DIR="{{EXPERIMENT_PATH}}"
  fi
  
  if [ -z "{{OUTPUT_PATH}}" ]; then
    python src/analysis/plot_results_over_time.py "${EXPERIMENT_DIR}" --over-time --cost-to-spend {{COST_TO_SPEND}}
  else
    python src/analysis/plot_results_over_time.py "${EXPERIMENT_DIR}" --over-time --cost-to-spend {{COST_TO_SPEND}} --output {{OUTPUT_PATH}}
  fi

# Plot test and validation performance over time for QuickTune experiments
# Automatically handles paths with or without seed_* directories
# Example: just plot-quicktune experiments/Cluster/QuickTune/test_metalearning_from_desmoid-liver_1e1c0a9_12-17-25_no-ftpfn
# Example with seed: just plot-quicktune experiments/Cluster/QuickTune/test_metalearning_from_desmoid-liver_1e1c0a9_12-17-25_no-ftpfn/seed_42
# Example with custom output: just plot-quicktune experiments/Cluster/QuickTune/test_experiment output.png
# Example over time (hours): just plot-quicktune-over-time experiments/Cluster/QuickTune/test_experiment
plot-quicktune EXPERIMENT_PATH OUTPUT_PATH="":
  #!/usr/bin/env bash
  # If path ends with seed_*, go up one level to get experiment directory
  if [[ "{{EXPERIMENT_PATH}}" == */seed_* ]]; then
    EXPERIMENT_DIR=$(dirname "{{EXPERIMENT_PATH}}")
  else
    EXPERIMENT_DIR="{{EXPERIMENT_PATH}}"
  fi
  
  if [ -z "{{OUTPUT_PATH}}" ]; then
    python src/analysis/plot_results_over_time.py "${EXPERIMENT_DIR}"
  else
    python src/analysis/plot_results_over_time.py "${EXPERIMENT_DIR}" --output {{OUTPUT_PATH}}
  fi

# Plot test and validation performance over time (wall-clock time) for QuickTune experiments
# Example: just plot-quicktune-over-time experiments/Cluster/QuickTune/test_metalearning_from_desmoid-liver_1e1c0a9_12-17-25_no-ftpfn
# Example with seed: just plot-quicktune-over-time experiments/Cluster/QuickTune/test_experiment/seed_42
# Example with custom output: just plot-quicktune-over-time experiments/Cluster/QuickTune/test_experiment output.png
plot-quicktune-over-time EXPERIMENT_PATH OUTPUT_PATH="":
  #!/usr/bin/env bash
  # If path ends with seed_*, go up one level to get experiment directory
  if [[ "{{EXPERIMENT_PATH}}" == */seed_* ]]; then
    EXPERIMENT_DIR=$(dirname "{{EXPERIMENT_PATH}}")
  else
    EXPERIMENT_DIR="{{EXPERIMENT_PATH}}"
  fi
  
  if [ -z "{{OUTPUT_PATH}}" ]; then
    python src/analysis/plot_results_over_time.py "${EXPERIMENT_DIR}" --over-time
  else
    python src/analysis/plot_results_over_time.py "${EXPERIMENT_DIR}" --over-time --output {{OUTPUT_PATH}}
  fi

# Plot test and validation performance over time for multiple experiments together
# Example: just plot-performance-over-time-multi test_plot experiments/NePS/lipo/test_plotting_script experiments/NePS/lipo/test_plotting_script_2
# This will save plots to experiments/Plots/test_plot.png and experiments/Plots/test_plot.pdf
plot-performance-over-time-multi PLOT_NAME *EXPERIMENT_DIRS:
  #!/usr/bin/env bash
  OUTPUT_DIR="experiments/Plots"
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_PATH="${OUTPUT_DIR}/{{PLOT_NAME}}.png"
  python src/analysis/plot_results_over_time.py {{EXPERIMENT_DIRS}} --output "${OUTPUT_PATH}"

# Plot test and validation performance over time (wall-clock time) for multiple experiments together
# Baseline experiments (1 config) will show the same performance for all 24 hours
# NePS experiments will show the best validation performance at each hour
# Example: just plot-neps-over-time-multi lipo_comparison 86400 experiments/Cluster/NePS/lipo/random-search_full-search-space_lipo_l40 experiments/Cluster/Baseline/lipo/densenet_baseline_lipo_l40 experiments/Cluster/Baseline/lipo/resnet_baseline_lipo_l40
# Example with y-axis limits: just plot-neps-over-time-multi lipo_zoom 86400 Y_MIN=60 Y_MAX=90 experiments/Cluster/NePS/lipo/random-search_full-search-space_lipo_l40
# This will save plots to experiments/Plots/lipo_comparison.png and experiments/Plots/lipo_comparison.pdf
plot-runs-over-time-multi PLOT_NAME COST_TO_SPEND *EXPERIMENT_DIRS:
  #!/usr/bin/env bash
  OUTPUT_DIR="experiments/Plots"
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_PATH="${OUTPUT_DIR}/{{PLOT_NAME}}.png"
  
  # Parse optional Y_MIN and Y_MAX from EXPERIMENT_DIRS
  # They may appear as Y_MIN=value or Y_MAX=value in the arguments
  Y_MIN_VAL=""
  Y_MAX_VAL=""
  EXPERIMENT_DIRS_CLEAN=""
  
  for ARG in {{EXPERIMENT_DIRS}}; do
    if [[ "$ARG" == Y_MIN=* ]]; then
      Y_MIN_VAL="${ARG#Y_MIN=}"
    elif [[ "$ARG" == Y_MAX=* ]]; then
      Y_MAX_VAL="${ARG#Y_MAX=}"
    else
      EXPERIMENT_DIRS_CLEAN="$EXPERIMENT_DIRS_CLEAN $ARG"
    fi
  done
  
  # Build command with optional y-axis limits
  CMD="python src/analysis/plot_results_over_time.py${EXPERIMENT_DIRS_CLEAN} --over-time --cost-to-spend {{COST_TO_SPEND}} --output \"${OUTPUT_PATH}\""
  
  if [ -n "${Y_MIN_VAL}" ]; then
    CMD="${CMD} --y-min ${Y_MIN_VAL}"
  fi
  
  if [ -n "${Y_MAX_VAL}" ]; then
    CMD="${CMD} --y-max ${Y_MAX_VAL}"
  fi
  
  eval "${CMD}"


# Plot test and validation performance over time for multiple experiments together with extend flag
# Extends shorter experiments to match the longest one by repeating the last performance value
# Example: just plot-performance-over-time-multi-extend test_plot experiments/NePS/lipo/test_plotting_script experiments/Baseline/liver/test_liver_33
# This will save plots to experiments/Plots/test_plot.png and experiments/Plots/test_plot.pdf
plot-performance-over-time-multi-extend PLOT_NAME *EXPERIMENT_DIRS:
  #!/usr/bin/env bash
  OUTPUT_DIR="experiments/Plots"
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_PATH="${OUTPUT_DIR}/{{PLOT_NAME}}.png"
  python src/analysis/plot_results_over_time.py {{EXPERIMENT_DIRS}} --output "${OUTPUT_PATH}" --extend-to-max-configs

# Plot test and validation performance over time (number of configs) for NePS experiments on cluster (single experiment)
# Example: just plot-performance-over-time-cluster NePS/lipo/test_plotting_script
# Example with custom output: just plot-performance-over-time-cluster NePS/lipo/test_plotting_script experiments/Plots/output.png
plot-performance-over-time-cluster DATASET EXPERIMENT_NAME EXPERIMENT_DIR OUTPUT_PATH="":
  #!/usr/bin/env bash
  SEED=42
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/Baseline/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/
  sbatch --exclude=dlcgpu19 \
    --output=/work/dlclarge1/wagnerd-medquicktune/experiments/Baseline/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/%x.%A.%a.%N.err_out \
    --error=/work/dlclarge1/wagnerd-medquicktune/experiments/Baseline/{{DATASET}}/{{EXPERIMENT_NAME}}/seed_${SEED}/cluster_oe/%x.%A.%a.%N.err_out \
    --export=EXPERIMENT_DIR="{{EXPERIMENT_DIR}}",OUTPUT_PATH="{{OUTPUT_PATH}}" \
    cluster_scripts/plot_performance_over_time.sh

# Plot test and validation performance over time for multiple experiments together on cluster with extend flag
# Example: just plot-performance-over-time-multi-extend-cluster gist_baseline_vs_autonorm experiments/Baseline/gist/50epochs_stratisfied-cv_densenet-model experiments/Baseline/gist/50epochs_stratisfied-cv_densenet-model
plot-performance-over-time-multi-cluster PLOT_NAME *EXPERIMENT_DIRS:
  #!/usr/bin/env bash
  BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
  mkdir -p "${BASE_DIR}/experiments/Plots/cluster_oe/"
  cd "${BASE_DIR}/code/MedQuickTune"
  sbatch --exclude=dlcgpu19 \
    --output="${BASE_DIR}/experiments/Plots/cluster_oe/%x.%A.%a.%N.err_out" \
    --error="${BASE_DIR}/experiments/Plots/cluster_oe/%x.%A.%a.%N.err_out" \
    --export=PLOT_NAME="{{PLOT_NAME}}",EXPERIMENT_DIRS="{{EXPERIMENT_DIRS}}" \
    cluster_scripts/plot_performance_over_time_multi.sh

# Plot test and validation performance over time for multiple experiments together on cluster with extend flag
# Example: just plot-performance-over-time-multi-extend-cluster gist_baseline_vs_autonorm experiments/Baseline/gist/50epochs_stratisfied-cv_densenet-model experiments/Baseline/gist/50epochs_stratisfied-cv_densenet-model
plot-performance-over-time-multi-extend-cluster PLOT_NAME *EXPERIMENT_DIRS:
  #!/usr/bin/env bash
  BASE_DIR="/work/dlclarge1/wagnerd-medquicktune"
  mkdir -p "${BASE_DIR}/experiments/Plots/cluster_oe/"
  cd "${BASE_DIR}/code/MedQuickTune"
  sbatch --exclude=dlcgpu19 \
    --output="${BASE_DIR}/experiments/Plots/cluster_oe/%x.%A.%a.%N.err_out" \
    --error="${BASE_DIR}/experiments/Plots/cluster_oe/%x.%A.%a.%N.err_out" \
    --export=PLOT_NAME="{{PLOT_NAME}}",EXPERIMENT_DIRS="{{EXPERIMENT_DIRS}}" \
    cluster_scripts/plot_performance_over_time_multi_extend.sh

# Plot test and validation performance over time (wall-clock time) for multiple experiments together
# Baseline experiments (1 config) will show the same performance for all 24 hours
# NePS experiments will show the best validation performance at each hour
# Example: just plot-neps-over-time-multi lipo_comparison 86400 experiments/Cluster/NePS/lipo/random-search_full-search-space_lipo_l40 experiments/Cluster/Baseline/lipo/densenet_baseline_lipo_l40 experiments/Cluster/Baseline/lipo/resnet_baseline_lipo_l40
# Example with y-axis limits: just plot-neps-over-time-multi lipo_zoom 86400 Y_MIN=60 Y_MAX=90 experiments/Cluster/NePS/lipo/random-search_full-search-space_lipo_l40
# This will save plots to experiments/Plots/lipo_comparison.png and experiments/Plots/lipo_comparison.pdf
plot-runs-over-time-multi_new TITLE PLOT_NAME COST_TO_SPEND *EXPERIMENT_DIRS:
  #!/usr/bin/env bash
  OUTPUT_DIR="experiments/Plots"
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_PATH="${OUTPUT_DIR}/{{PLOT_NAME}}.png"

  # Parse optional Y_MIN and Y_MAX from EXPERIMENT_DIRS
  # They may appear as Y_MIN=value or Y_MAX=value in the arguments
  Y_MIN_VAL=""
  Y_MAX_VAL=""
  EXPERIMENT_DIRS_CLEAN=""

  for ARG in {{EXPERIMENT_DIRS}}; do
    if [[ "$ARG" == Y_MIN=* ]]; then
      Y_MIN_VAL="${ARG#Y_MIN=}"
    elif [[ "$ARG" == Y_MAX=* ]]; then
      Y_MAX_VAL="${ARG#Y_MAX=}"
    else
      EXPERIMENT_DIRS_CLEAN="$EXPERIMENT_DIRS_CLEAN $ARG"
    fi
  done
  
  # Build command with optional y-axis limits
  TITLE_VAL="{{TITLE}}"
  CMD="python src/analysis/plot_results_over_time_new.py${EXPERIMENT_DIRS_CLEAN} --title \"${TITLE_VAL}\" --over-time --cost-to-spend {{COST_TO_SPEND}} --output \"${OUTPUT_PATH}\""

  if [ -n "${Y_MIN_VAL}" ]; then
    CMD="${CMD} --y-min ${Y_MIN_VAL}"
  fi

  if [ -n "${Y_MAX_VAL}" ]; then
    CMD="${CMD} --y-max ${Y_MAX_VAL}"
  fi

  eval "${CMD}"
