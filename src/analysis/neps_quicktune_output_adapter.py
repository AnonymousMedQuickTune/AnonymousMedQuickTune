#!/usr/bin/env python3
"""
NePS to QuickTune Output Adapter

This script converts NePS optimization results into the format required by QuickTune.
It creates four CSV files:
- config.csv: Hyperparameter configurations
- curve.csv: Learning curves
- cost.csv: Runtime costs
- meta.csv: Meta-features of the dataset
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pandas as pd
import yaml

import argparse
import ast
import os
from typing import Any, Dict, List, Tuple

import numpy as np


class NePSQuickTuneAdapter:
    """
    Adapter class to convert NePS optimization results into QuickTune format.

    This class handles the conversion of NePS output files into four CSV files
    required by QuickTune:
    - config.csv: Contains hyperparameter configurations
    - curve.csv: Contains learning curves data
    - cost.csv: Contains runtime costs
    - meta.csv: Contains meta-features of the dataset

    The adapter parses the NePS output, transforms the data into appropriate
    formats, and saves the results as CSV files in the specified output directory.
    """

    def __init__(self, input_path: str, output_dir: str):
        """
        Initialize the adapter with input and output paths.

        Args:
            input_path: Path to the experiment directory (e.g., experiments/brain_tumor/test_8/seed_42)
            output_dir: Directory where the CSV files should be saved
        """
        self.input_path = Path(input_path)
        self.neps_output = (
            self.input_path / "NePS_output"
        )  # Add NePS_output to path when needed
        self.output_dir = Path(output_dir)
        self.setup_logging()

        # Read hydra config and store target_metric
        config_path = os.path.join(
            self.input_path,  # Already at experiment dir level
            "hydra_output/config.yaml",
        )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                hydra_config = yaml.safe_load(f)
                self.target_metric = hydra_config["metric"]
        except FileNotFoundError:
            logging.error(f"Hydra config not found at: {config_path}")
            raise

    @staticmethod
    def setup_logging() -> None:
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def parse_neps_output(self) -> List[Dict[str, Any]]:
        """Parse the NePS summary CSV file into a list of dictionaries."""
        # First, load the hydra config to get model type and dataset
        config_path = os.path.join(self.input_path, "hydra_output/config.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                hydra_config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Hydra config not found at: {config_path}")
            raise

        results = []
        summary_path = os.path.join(self.neps_output, "summary", "full.csv")

        try:
            df = pd.read_csv(summary_path)

            for _, row in df.iterrows():
                config_dict = {}

                # Extract all configuration parameters
                for column in df.columns:
                    if column.startswith("config."):
                        param_name = column.replace("config.", "")
                        value = row[column]
                        # Convert to int if the parameter name suggests it should be an integer
                        if any(
                            int_param in param_name
                            for int_param in ["epochs", "batch_size"]
                        ):
                            value = int(value)
                        config_dict[param_name] = value

                # Add model type and dataset info
                config_dict["model_type"] = hydra_config["model"]["type"]
                config_dict["dataset"] = hydra_config["data"]["dataset"]

                # Add performance metrics
                config_dict["final_accuracy"] = -row[
                    "objective_to_minimize"
                ]  # Assuming objective is negative accuracy

                # Add learning curve if available
                if "learning_curve" in row:
                    try:
                        config_dict["curves"] = ast.literal_eval(row["learning_curve"])
                    except (SyntaxError, ValueError):
                        config_dict["curves"] = []

                results.append(config_dict)

            logging.info("Successfully parsed %d configurations", len(results))
            return results

        except FileNotFoundError:
            logging.error(f"NePS summary file not found at: {summary_path}")
            raise
        except Exception as e:
            logging.error(f"Error parsing NePS summary file: {str(e)}")
            raise

    def create_dataframes(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create DataFrames for configurations, learning curves, costs, and meta-features."""

        def read_hydra_config() -> dict:
            """Helper function to read hydra config."""
            config_path = os.path.join(self.input_path, "hydra_output/config.yaml")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                logging.error(f"Hydra config not found at: {config_path}")
                raise

        hydra_config = read_hydra_config()

        # Create meta-features DataFrame with dataset first, then known values for brain tumor dataset
        meta_data = {
            "dataset": [],
            "num_classes": [],
            "input_channels": [],
            "input_size": [],
            "total_train_samples": [],
            "total_val_samples": [],
            "total_test_samples": [],
        }

        # Fill meta-features with dataset name and known values
        for _ in results:
            meta_data["dataset"].append(hydra_config["data"]["dataset"])
            meta_data["num_classes"].append(4)  # 4 classes in brain tumor dataset
            meta_data["input_channels"].append(3)  # RGB images
            meta_data["input_size"].append(224)  # 224x224 images
            meta_data["total_train_samples"].append(2513)  # Training set size
            meta_data["total_val_samples"].append(628)  # Validation set size
            meta_data["total_test_samples"].append(785)  # Test set size

        meta_df = pd.DataFrame(meta_data)

        # Create configurations DataFrame with model_type and dataset first
        config_data = {"model_type": [], "dataset": []}

        # Add all other hyperparameters
        for key in results[0].keys():
            if key not in ["curves", "final_accuracy", "model_type", "dataset"]:
                config_data[key] = []

        # Fill the data
        for result in results:
            config_data["model_type"].append(result["model_type"])
            config_data["dataset"].append(result["dataset"])
            for key in config_data.keys():
                if key not in ["model_type", "dataset"]:
                    config_data[key].append(result[key])

        config_df = pd.DataFrame(config_data)

        # Create learning curves DataFrame by reading validation metrics from each fold
        curves_data = []

        for idx, result in enumerate(results, start=1):  # Start enumeration at 1
            # Updated path: use self.neps_output which includes NePS_output
            config_dir = os.path.join(self.neps_output, "configs", f"config_{idx}_0")

            try:
                # Get number of folds by counting fold directories
                k_folds = len(
                    [
                        d
                        for d in os.listdir(config_dir)
                        if d.startswith("fold_")
                        and os.path.isdir(os.path.join(config_dir, d))
                    ]
                )

                # Read metrics from each fold
                fold_curves = self.get_fold_metrics(idx, k_folds)  # idx is now 1-based

                if fold_curves:
                    # Ensure all folds have same number of epochs
                    min_epochs = min(len(curve) for curve in fold_curves)
                    fold_curves = [curve[:min_epochs] for curve in fold_curves]

                    # Average over folds for each epoch
                    fold_curves = np.array(fold_curves)
                    avg_curve = np.mean(fold_curves, axis=0)
                    curves_data.append(avg_curve)
                else:
                    logging.warning(f"No valid curves found for config {idx}")
                    curves_data.append(np.zeros(1))

            except FileNotFoundError:
                logging.warning(f"Config directory not found: {config_dir}")
                curves_data.append(np.zeros(1))

        curves_df = pd.DataFrame(curves_data)

        # Create cost DataFrame using number_of_epochs as proxy for cost
        cost_data = {"cost": [r["number_of_epochs"] for r in results]}
        cost_df = pd.DataFrame(cost_data)

        return config_df, curves_df, cost_df, meta_df

    def get_fold_metrics(self, config_idx: int, k_folds: int) -> List[np.ndarray]:
        """Helper function to read metrics from each fold."""
        fold_curves = []
        # Updated path: use self.neps_output which includes NePS_output
        config_dir = os.path.join(self.neps_output, "configs", f"config_{config_idx}_0")

        for fold in range(k_folds):
            metrics_path = os.path.join(
                config_dir, f"fold_{fold}", "logging", "metrics.csv"
            )
            try:
                metrics_df = pd.read_csv(metrics_path)
                val_metrics = metrics_df[metrics_df["phase"] == "val"][
                    self.target_metric
                ].values
                fold_curves.append(val_metrics)
            except FileNotFoundError:
                logging.warning(
                    f"Metrics file not found for config {config_idx}, fold {fold}"
                )
            except KeyError:
                logging.error(
                    f"Metric '{self.target_metric}' not found in columns: {metrics_df.columns}"
                )
                raise
        return fold_curves

    def save_dataframes(
        self,
        config_df: pd.DataFrame,
        curves_df: pd.DataFrame,
        cost_df: pd.DataFrame,
        meta_df: pd.DataFrame,
    ) -> None:
        """
        Save DataFrames to CSV files.

        Args:
            config_df: Configuration DataFrame
            curves_df: Learning curves DataFrame
            cost_df: Cost DataFrame
            meta_df: Meta-features DataFrame
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Add 1-based indices to match the config IDs
            config_df.index = range(1, len(config_df) + 1)
            curves_df.index = range(1, len(curves_df) + 1)
            cost_df.index = range(1, len(cost_df) + 1)
            meta_df.index = range(1, len(meta_df) + 1)

            # Save with index=True to include the config IDs
            config_df.to_csv(self.output_dir / "config.csv", index=True)
            curves_df.to_csv(self.output_dir / "curve.csv", index=True)
            cost_df.to_csv(self.output_dir / "cost.csv", index=True)
            meta_df.to_csv(self.output_dir / "meta.csv", index=True)

            logging.info("Successfully saved CSV files to %s", self.output_dir)
        except Exception as e:
            logging.error("Failed to save CSV files: %s", str(e))
            raise

    def convert(self) -> None:
        """Execute the complete conversion process."""
        logging.info("Starting conversion from %s", self.input_path)
        results = self.parse_neps_output()
        config_df, curves_df, cost_df, meta_df = self.create_dataframes(results)
        self.save_dataframes(config_df, curves_df, cost_df, meta_df)
        logging.info("Conversion completed successfully")


def merge_neps_runs(
    base_dir: str | Path,
    experiment_names: List[str],
    seeds: List[str],
    output_dir: str | Path,
) -> None:
    """
    Merge multiple NePS runs into a single portfolio directory.

    Args:
        base_dir: Base directory containing all experiments (e.g., experiments/brain_tumor)
        experiment_names: List of experiment names to merge (can contain duplicates for multiple seeds)
        seeds: List of seeds to merge (must match length of experiment_names)
        output_dir: Directory to save the merged portfolio

    Raises:
        ValueError: If no valid NePS runs are found to merge
        FileNotFoundError: If specified directories don't exist
    """
    all_configs = []
    all_curves = []
    all_costs = []
    all_meta = []

    base_path = Path(base_dir)
    processed_runs = set()  # Track which runs we've already processed

    # Verify lengths match
    if len(experiment_names) != len(seeds):
        raise ValueError(f"Number of experiments ({len(experiment_names)}) must match number of seeds ({len(seeds)})")

    # Process each experiment-seed pair
    for exp_name, seed in zip(experiment_names, seeds):
        run_id = f"{exp_name}_{seed}"
        if run_id in processed_runs:
            continue
        processed_runs.add(run_id)

        # Updated path: use experiment directory path
        exp_dir = base_path / exp_name / f"seed_{seed}"

        # Debug logging
        logging.info(f"Looking for experiment at: {exp_dir}")
        if exp_dir.exists():
            logging.info(f"Found experiment at: {exp_dir}")
        else:
            logging.warning(f"No experiment found at {exp_dir}")
            continue

        adapter = NePSQuickTuneAdapter(str(exp_dir), output_dir)
        results = adapter.parse_neps_output()
        config_df, curves_df, cost_df, meta_df = adapter.create_dataframes(results)

        all_configs.append(config_df)
        all_curves.append(curves_df)
        all_costs.append(cost_df)
        all_meta.append(meta_df)

    if not all_configs:
        raise ValueError("No valid NePS runs found to merge")

    # Merge all dataframes
    merged_config = pd.concat(all_configs, ignore_index=True)
    merged_curves = pd.concat(all_curves, ignore_index=True)
    merged_costs = pd.concat(all_costs, ignore_index=True)
    merged_meta = pd.concat(all_meta, ignore_index=True)

    # Create portfolio directory
    portfolio_dir = Path(output_dir)
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    # Add 1-based indices to match the config IDs
    merged_config.index = range(1, len(merged_config) + 1)
    merged_curves.index = range(1, len(merged_curves) + 1)
    merged_costs.index = range(1, len(merged_costs) + 1)
    merged_meta.index = range(1, len(merged_meta) + 1)

    # Save merged files with index
    merged_config.to_csv(portfolio_dir / "config.csv", index=True)
    merged_curves.to_csv(portfolio_dir / "curve.csv", index=True)
    merged_costs.to_csv(portfolio_dir / "cost.csv", index=True)
    merged_meta.to_csv(portfolio_dir / "meta.csv", index=True)

    logging.info(
        f"Successfully merged {len(processed_runs)} NePS runs into {portfolio_dir}"
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="main_experiment_config.yaml",
)
def main(config: DictConfig) -> None:  # TODO: adapt script for multiple datasets
    """Main entry point of the script."""
    try:
        if config.get("merge_runs", False):
            # Get experiment names and seeds from config
            experiment_names = config.get("experiment_names", "").split(",")
            seeds = config.get("seeds", "").split(",")
            
            if not experiment_names or not seeds:
                raise ValueError(
                    "Both experiment_names and seeds must be specified in config when merging runs"
                )
                
            merge_neps_runs(
                base_dir=config.base_dir,
                experiment_names=experiment_names,
                seeds=seeds,
                output_dir=config.portfolio_dir
            )
        else:
            adapter = NePSQuickTuneAdapter(
                input_path=config.experiment_base_dir,
                output_dir=config.quicktune_directory
            )
            adapter.convert()
            
    except Exception as e:
        logging.error("Conversion failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
