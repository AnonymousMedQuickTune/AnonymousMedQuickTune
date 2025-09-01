#!/usr/bin/env python3
"""
Portfolio Creation Script

This script creates QuickTune portfolios by merging multiple NePS experiments from different datasets.
It creates four CSV files:
- config.csv: Hyperparameter configurations
- curve.csv: Learning curves
- cost.csv: Runtime costs
- meta.csv: Meta-features of the dataset
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig


# Constants
CONFIG_PREFIX = "config."
HYDRA_CONFIG_FILE = "experimental_setting.yaml"
NEPS_OUTPUT_DIR = "NePS_output"
HYDRA_OUTPUT_DIR = "hydra_output"
SUMMARY_FILE = "full.csv"
METRICS_FILE = "metrics.csv"
CONFIG_DIR_PREFIX = "config_"
FOLD_DIR_PREFIX = "fold_"
CV_FOLD_DIR = "cv_fold_0"

# Default meta-features (can be overridden by dataset-specific configs)
DEFAULT_META_FEATURES = {
    "num_classes": 2,
    "input_channels": 1,
    "input_size": 224,
    "total_train_samples": 1000,
    "total_val_samples": 200,
    "total_test_samples": 300,
}

# Integer parameter names that should be converted to int
INT_PARAM_NAMES = {"epochs", "batch_size"}

# Parsing constants
DATASET_SEPARATOR = ';'
EXPERIMENT_SEPARATOR = ','
SEED_SEPARATOR = ','
PARENTHESIS_OPEN = '('
PARENTHESIS_CLOSE = ')'
DATASET_EXPERIMENT_SEPARATOR = ':'

# Epoch keys to try for cost calculation
EPOCH_KEYS = ["number_of_epochs", "epochs", "num_epochs", "training_epochs"]

# Base paths
EXPERIMENTS_BASE_PATH = "experiments/NePS"

# Directory structure patterns
SUMMARY_SUBDIR = "summary"
CONFIGS_SUBDIR = "configs"
LOGGING_SUBDIR = "logging"

# Output file names
CONFIG_CSV = "config.csv"
CURVE_CSV = "curve.csv"
COST_CSV = "cost.csv"
META_CSV = "meta.csv"


class PortfolioCreator:
    """
    Portfolio creator class to convert NePS optimization results into QuickTune format.

    This class handles the conversion of NePS output files into four CSV files
    required by QuickTune for portfolio creation.
    """

    def __init__(self, input_path: str, output_dir: str):
        """
        Initialize the portfolio creator with input and output paths.

        Args:
            input_path: Path to the experiment directory (e.g., experiments/lipo/test_portfolio_1/seed_42)
            output_dir: Directory where the CSV files should be saved
        """
        self.input_path = Path(input_path)
        self.neps_output = self.input_path / NEPS_OUTPUT_DIR
        self.output_dir = Path(output_dir)
        self.setup_logging()
        
        # Load configuration
        self.hydra_config = self._load_hydra_config()
        self.target_metric = self.hydra_config["metric"]
    
    def _load_hydra_config(self) -> Dict[str, Any]:
        """Load Hydra configuration from experimental_setting.yaml."""
        config_path = self.input_path / HYDRA_OUTPUT_DIR / HYDRA_CONFIG_FILE
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def setup_logging() -> None:
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def parse_neps_output(self) -> List[Dict[str, Any]]:
        """Parse the NePS summary CSV file into a list of dictionaries."""
        summary_path = self._find_summary_file()
        df = pd.read_csv(summary_path)
        results = []

        for _, row in df.iterrows():
            config_dict = self._extract_config_from_row(row, df.columns)
            results.append(config_dict)

        logging.info("Successfully parsed %d configurations", len(results))
        return results
    
    def _find_summary_file(self) -> Path:
        """Find the NePS summary file, checking new structure first."""
        # Check for new outer fold structure first, then fall back to old structure
        summary_path = self.neps_output / CV_FOLD_DIR / SUMMARY_SUBDIR / SUMMARY_FILE
        if not summary_path.exists():
            summary_path = self.neps_output / SUMMARY_SUBDIR / SUMMARY_FILE
        return summary_path
    
    def _extract_config_from_row(self, row: pd.Series, columns: List[str]) -> Dict[str, Any]:
        """Extract configuration from a single row of the summary CSV."""
        config_dict = {}

        # Extract all configuration parameters
        for column in columns:
            if column.startswith(CONFIG_PREFIX):
                param_name = column.replace(CONFIG_PREFIX, "")
                value = row[column]
                # Convert to int if the parameter name suggests it should be an integer
                if any(int_param in param_name for int_param in INT_PARAM_NAMES):
                    value = int(value)
                config_dict[param_name] = value

        # Add model type and dataset info
        config_dict["model_type"] = self.hydra_config["model"]["type"]
        config_dict["dataset"] = self.hydra_config["data"]["dataset"]

        # Add performance metrics
        config_dict["final_accuracy"] = -row["objective_to_minimize"]

        # Add learning curve if available
        if "learning_curve" in row:
            config_dict["curves"] = ast.literal_eval(row["learning_curve"])

        return config_dict

    def create_dataframes(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create DataFrames for configurations, learning curves, costs, and meta-features."""
        if not results:
            raise ValueError("No results provided to create dataframes")
        
        config_df = self._create_config_dataframe(results)
        curves_df = self._create_curves_dataframe(results)
        cost_df = self._create_cost_dataframe(results)
        meta_df = self._create_meta_dataframe(results)
        
        return config_df, curves_df, cost_df, meta_df
    
    def _create_config_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create configuration DataFrame."""
        # Get all parameter keys except excluded ones
        excluded_keys = {"curves", "final_accuracy", "model_type", "dataset"}
        param_keys = [key for key in results[0].keys() if key not in excluded_keys]
        
        # Build config data
        config_data = {}
        for result in results:
            for key in ["model_type", "dataset"] + param_keys:
                if key not in config_data:
                    config_data[key] = []
                config_data[key].append(result[key])

        return pd.DataFrame(config_data)
    
    def _create_meta_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create meta-features DataFrame."""
        meta_features = self._get_dataset_meta_features()
        dataset_name = self.hydra_config["data"]["dataset"]
        
        # Create meta data for all results
        meta_data = {
            "dataset": [dataset_name] * len(results),
            "num_classes": [meta_features["num_classes"]] * len(results),
            "input_channels": [meta_features["input_channels"]] * len(results),
            "input_size": [meta_features["input_size"]] * len(results),
            "total_train_samples": [meta_features["total_train_samples"]] * len(results),
            "total_val_samples": [meta_features["total_val_samples"]] * len(results),
            "total_test_samples": [meta_features["total_test_samples"]] * len(results),
        }

        return pd.DataFrame(meta_data)
    
    def _get_dataset_meta_features(self) -> Dict[str, int]:
        """Get dataset-specific meta-features from config or return defaults."""
        dataset = self.hydra_config["data"]["dataset"]
        
        # Try to get meta-features from config first
        if "meta_features" in self.hydra_config.get("data", {}):
            config_meta = self.hydra_config["data"]["meta_features"]
            return {
                "num_classes": config_meta.get("num_classes", DEFAULT_META_FEATURES["num_classes"]),
                "input_channels": config_meta.get("input_channels", DEFAULT_META_FEATURES["input_channels"]),
                "input_size": config_meta.get("input_size", DEFAULT_META_FEATURES["input_size"]),
                "total_train_samples": config_meta.get("total_train_samples", DEFAULT_META_FEATURES["total_train_samples"]),
                "total_val_samples": config_meta.get("total_val_samples", DEFAULT_META_FEATURES["total_val_samples"]),
                "total_test_samples": config_meta.get("total_test_samples", DEFAULT_META_FEATURES["total_test_samples"]),
            }
        
        # Fallback to defaults
        return DEFAULT_META_FEATURES
    
    def _create_cost_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create cost DataFrame."""
        # Try different possible keys for epochs
        epoch_key = next((key for key in EPOCH_KEYS if key in results[0]), None)
        
        if epoch_key:
            cost_data = {"cost": [r[epoch_key] for r in results]}
        else:
            logging.warning("No epoch key found, using default cost of 1")
            cost_data = {"cost": [1 for _ in results]}
        
        return pd.DataFrame(cost_data)
    
    def _create_curves_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create learning curves DataFrame."""
        curves_data = []

        for idx, result in enumerate(results, start=1):  # Start enumeration at 1
            config_dir = self._find_config_directory(idx)
            
            if not config_dir.exists():
                logging.warning(f"Config directory not found: {config_dir}")
                curves_data.append(np.zeros(1))
                continue
                
            k_folds = self._count_folds(config_dir)
            fold_curves = self.get_fold_metrics(idx, k_folds, str(config_dir))

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

        return pd.DataFrame(curves_data)
    
    def _find_config_directory(self, config_idx: int) -> Path:
        """Find the configuration directory, checking new structure first."""
        # Check for new outer fold structure first, then fall back to old structure
        config_dir = self.neps_output / CV_FOLD_DIR / CONFIGS_SUBDIR / f"{CONFIG_DIR_PREFIX}{config_idx}"
        if not config_dir.exists():
            config_dir = self.neps_output / CONFIGS_SUBDIR / f"{CONFIG_DIR_PREFIX}{config_idx}_0"
        return config_dir
    
    def _count_folds(self, config_dir: Path) -> int:
        """Count the number of fold directories in a config directory."""
        return sum(1 for d in config_dir.iterdir() 
                  if d.is_dir() and d.name.startswith(FOLD_DIR_PREFIX))

    def get_fold_metrics(self, config_idx: int, k_folds: int, config_dir: str = None) -> List[np.ndarray]:
        """Helper function to read metrics from each fold."""
        fold_curves = []
        
        # Use provided config_dir or construct it
        if config_dir is None:
            config_dir = str(self._find_config_directory(config_idx))

        for fold in range(k_folds):
            metrics_path = Path(config_dir) / f"{FOLD_DIR_PREFIX}{fold}" / LOGGING_SUBDIR / METRICS_FILE
            
            if not metrics_path.exists():
                logging.warning(f"Metrics file not found for config {config_idx}, fold {fold}")
                continue
                
            metrics_df = pd.read_csv(metrics_path)
            if self.target_metric not in metrics_df.columns:
                logging.error(f"Metric '{self.target_metric}' not found in columns: {metrics_df.columns}")
                continue
                
            val_metrics = metrics_df[metrics_df["phase"] == "val"][self.target_metric].values
            fold_curves.append(val_metrics)
            
        return fold_curves


def parse_experiment_seeds(experiment_spec: str) -> List[Tuple[str, str]]:
    """Parse experiment specification string into experiment-seed pairs."""
    if not experiment_spec or not experiment_spec.strip():
        raise ValueError("Experiment specification cannot be empty")
    
    experiments = _split_experiments(experiment_spec)
    pairs = []
    
    for exp in experiments:
        exp = exp.strip()
        if not exp:
            continue
            
        if PARENTHESIS_OPEN in exp and exp.endswith(PARENTHESIS_CLOSE):
            exp_name, seeds_str = _extract_experiment_and_seeds(exp)
            seeds = _parse_seeds(seeds_str)
            pairs.extend((exp_name, seed) for seed in seeds)
        else:
            pairs.append((exp, ""))
    
    return pairs


def parse_dataset_experiment_specs(dataset_spec: str) -> List[Tuple[str, str]]:
    """Parse dataset-experiment specification string into dataset-experiment pairs."""
    if not dataset_spec or not dataset_spec.strip():
        raise ValueError("Dataset specification cannot be empty")
    
    pairs = []
    for spec in dataset_spec.split(DATASET_SEPARATOR):
        spec = spec.strip()
        if not spec:
            continue
            
        if DATASET_EXPERIMENT_SEPARATOR not in spec:
            raise ValueError(
                f"Invalid dataset specification format: '{spec}'. "
                f"Expected 'dataset{DATASET_EXPERIMENT_SEPARATOR}experiment_spec'"
            )
        
        dataset, experiment_spec = spec.split(DATASET_EXPERIMENT_SEPARATOR, 1)
        pairs.append((dataset.strip(), experiment_spec.strip()))
    
    return pairs


def _split_experiments(experiment_spec: str) -> List[str]:
    """Split experiment specification by commas, respecting parentheses."""
    experiments = []
    current_exp = ""
    paren_count = 0
    
    for char in experiment_spec:
        if char == PARENTHESIS_OPEN:
            paren_count += 1
        elif char == PARENTHESIS_CLOSE:
            paren_count -= 1
        elif char == EXPERIMENT_SEPARATOR and paren_count == 0:
            experiments.append(current_exp.strip())
            current_exp = ""
            continue
        current_exp += char
    
    if current_exp.strip():
        experiments.append(current_exp.strip())
    
    return experiments


def _extract_experiment_and_seeds(exp: str) -> Tuple[str, str]:
    """Extract experiment name and seeds string from experiment specification."""
    exp_name, seeds_str = exp.split(PARENTHESIS_OPEN, 1)
    seeds_str = seeds_str.rstrip(PARENTHESIS_CLOSE)
    return exp_name.strip(), seeds_str


def _parse_seeds(seeds_str: str) -> List[str]:
    """Parse seeds string into list of individual seeds."""
    if not seeds_str.strip():
        return []
    
    seeds = [seed.strip() for seed in seeds_str.split(SEED_SEPARATOR)]
    for seed in seeds:
        if not seed.isdigit():
            raise ValueError(f"Invalid seed value: '{seed}'. Seeds must be numeric.")
    
    return seeds


def merge_neps_runs_multi_dataset(
    dataset_spec: str,
    output_dir: str | Path,
) -> None:
    """
    Merge multiple NePS runs from multiple datasets into a single portfolio directory.

    Args:
        dataset_spec: Dataset-experiment specification string 
                     (e.g., 'lipo:test_portfolio_1(42,43),test_portfolio_2(43,44);desmoid:test_portfolio_5(42,43),test_portfolio_2(43,44)')
        output_dir: Directory to save the merged portfolio

    Raises:
        ValueError: If no valid NePS runs are found to merge
        FileNotFoundError: If specified directories don't exist
    """
    all_configs = []
    all_curves = []
    all_costs = []
    all_meta = []

    processed_runs = set()  # Track which runs we've already processed

    # Parse dataset-experiment pairs
    dataset_experiment_pairs = parse_dataset_experiment_specs(dataset_spec)
    
    if not dataset_experiment_pairs:
        raise ValueError("No valid dataset-experiment pairs found in specification")
    
    logging.info(f"Parsed dataset-experiment pairs: {dataset_experiment_pairs}")

    # Process each dataset with its specific experiments
    for dataset, experiment_spec in dataset_experiment_pairs:
        base_path = Path(EXPERIMENTS_BASE_PATH) / dataset
        
        # Parse experiment-seed pairs for this dataset
        experiment_seed_pairs = parse_experiment_seeds(experiment_spec)
        logging.info(f"Dataset {dataset}: parsed experiment-seed pairs: {experiment_seed_pairs}")
        
        # Process each experiment-seed pair for this dataset
        for exp_name, seed in experiment_seed_pairs:
            if not seed:  # Skip if no seed specified
                continue
                
            run_id = f"{dataset}_{exp_name}_{seed}"
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

            adapter = PortfolioCreator(str(exp_dir), output_dir)
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
    merged_config.to_csv(portfolio_dir / CONFIG_CSV, index=True)
    merged_curves.to_csv(portfolio_dir / CURVE_CSV, index=True)
    merged_costs.to_csv(portfolio_dir / COST_CSV, index=True)
    merged_meta.to_csv(portfolio_dir / META_CSV, index=True)

    logging.info(
        f"Successfully merged {len(processed_runs)} NePS runs from {len(dataset_experiment_pairs)} datasets into {portfolio_dir}"
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="experimental_setting.yaml",
)
def main(config: DictConfig) -> None:
    """Main entry point for portfolio creation."""
    dataset_spec = config.get("dataset_spec", "")
    
    if not dataset_spec:
        raise ValueError(
            "dataset_spec must be specified. Use +dataset_spec='...' on command line"
        )
        
    merge_neps_runs_multi_dataset(
        dataset_spec=dataset_spec,
        output_dir=config.portfolio_dir
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter