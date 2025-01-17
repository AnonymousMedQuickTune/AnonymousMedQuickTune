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

# EXPERIMENTS --------------------------------------------------------------------------------------

# Run a test experiment on the local machine
run-local-test EXPERIMENT_NAME SEED:
  python -m src.train data.dataset=desmoid experiment_name={{EXPERIMENT_NAME}} seed={{SEED}}

# Submit a test experiment to the cluster
run-cluster-test EXPERIMENT_NAME SEED:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/
  sbatch --exclude=dlcgpu05 --output=/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge1/wagnerd-medquicktune/experiments/desmoid/{{EXPERIMENT_NAME}}/seed_{{SEED}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}},SEED={{SEED}} cluster_scripts/desmoid_test.sh
