#!/bin/bash

# Set the main directory (assumes the script is located in 'utils' under the main directory)
MAIN_DIR=$(dirname "$(dirname "$(realpath "$0")")")
EXPERIMENTS_DIR="$MAIN_DIR/experiments"

echo "Deleting all experiments starting with 'test' in directory: $EXPERIMENTS_DIR"

# Check if the 'experiments' directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
  echo "The 'experiments' directory does not exist. Exiting."
  exit 1
fi

# Iterate over NePS and QuickTune directories
for framework_dir in "$EXPERIMENTS_DIR"/{NePS,QuickTune}; do
  # Check if the framework directory exists
  if [ -d "$framework_dir" ]; then
    echo "Checking framework directory: $(basename "$framework_dir")"
    
    # Iterate over all dataset directories within the framework directory
    for dataset_dir in "$framework_dir"/*; do
      # Check if the current item is a directory
      if [ -d "$dataset_dir" ]; then
        echo "Checking experiments in dataset directory: $(basename "$dataset_dir")"

        # Find all directories inside the dataset directory that start with 'test' or 'Test'
        for experiment_dir in "$dataset_dir"/[tT]est*; do
          # Check if the matched item is a directory (to avoid non-directory matches)
          if [ -d "$experiment_dir" ]; then
            echo "Deleting experiment: $(basename "$experiment_dir")"
            rm -rf "$experiment_dir"  # Delete the experiment directory
          fi
        done
      fi
    done
  fi
done

echo "All matching experiments have been deleted."
