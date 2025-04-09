import argparse
import os
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.test_best_config import test_run_pipeline
from src.analysis.generalization_analysis import (
    analyze_training_validation_metrics,
    analyze_validation_test_generalization
)
import traceback


def parse_neps_results(neps_output_dir: str):
    """
    Parse the results from NePS summary CSV file.

    Args:
        neps_output_dir (str): Path to NePS output directory

    Returns:
        tuple: (dict, str) - (best hyperparameter config, config ID)
    """
    # Read the summary CSV
    summary_path = os.path.join(neps_output_dir, "summary", "full.csv")
    df = pd.read_csv(summary_path)
    
    # Find the best configuration (minimum objective_to_minimize)
    best_row = df.loc[df['objective_to_minimize'].idxmin()]
    
    # Extract config parameters dynamically
    config_params = {}
    for column in df.columns:
        if column.startswith('config.'):
            param_name = column.replace('config.', '')
            value = best_row[column]
            # Convert to int if the parameter name suggests it should be an integer
            if any(int_param in param_name for int_param in ['epochs', 'batch_size']):
                value = int(value)
            config_params[param_name] = value
    
    # Get config ID from the 'id' column
    config_id = best_row['id']
    
    print("\nBest configuration found:")
    print(f"Config ID: {config_id}")
    print("Parameters:", config_params)
    print(f"Performance: {-best_row['objective_to_minimize']:.2f}%\n")
    
    return config_params, config_id


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NePS optimization results"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., experiments/brain_tumor/test_7/seed_42)",
    )
    parser.add_argument(
        "--hydra_config",
        type=str,
        required=True,
        help="Path to hydra config file (e.g., configs/main_experiment_config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to override config",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        required=True,
        help="Number of folds for cross-validation",
    )
    args = parser.parse_args()

    # Load the hydra config
    config = OmegaConf.load(args.hydra_config)

    # Override the dataset and data path in config
    config.data.dataset = args.dataset
    config.data.path = args.data_dir

    # Get NePS output directory
    neps_output_dir = os.path.join(args.experiment_dir, "NePS_output")
    
    # Get the best hyperparameters and config ID
    best_hyperparameters, config_id = parse_neps_results(neps_output_dir)

    # Analyze generalization across all configurations
    analyze_training_validation_metrics(neps_output_dir, args.k_folds)

    # Create test directory
    test_dir = Path(config.experiment_base_dir) / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Run test evaluation with best configuration
    avg_metrics = test_run_pipeline(
        _pipeline_directory=str(test_dir),
        _previous_pipeline_directory=None,
        config=config,
        neps_output_dir=neps_output_dir,
        config_id=config_id,
        k_folds=args.k_folds,
        **best_hyperparameters,
    )

    # Analyze validation-test generalization
    analyze_validation_test_generalization(neps_output_dir, avg_metrics, args.k_folds)


if __name__ == "__main__":
    main()
