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
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig

from src.utils.quicktune_utils import custom_extract_image_dataset_metafeat


# Constants
CONFIG_PREFIX = "config."
HYDRA_CONFIG_FILE = "experimental_setting.yaml"
NEPS_OUTPUT_DIR = "NePS_output"
HYDRA_OUTPUT_DIR = "hydra_output"
SUMMARY_FILE = "full.csv"
METRICS_FILE = "metrics.csv"
CONFIG_DIR_PREFIX = "config_"
FOLD_DIR_PREFIX = "cv_inner_fold_"
CV_FOLD_DIR = "cv_outer_fold_0"

# Default meta-features (can be overridden by dataset-specific configs)
DEFAULT_META_FEATURES = {
    "num_classes": 2,
    "input_channels": 1,
    "input_size_h": 224,
    "input_size_w": 224,
    "input_size_d": 224,
    "modality": "CT",  # Default to CT
    "total_num_samples": 1500,  # Total number of samples (matches quicktune_utils.py format)
}

# Mapping of dataset names to modality
MRI_DATASETS = ["lipo", "desmoid", "liver"]
CT_DATASETS = ["brain_tumor"]  # Add other CT datasets as needed

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
import os
EXPERIMENTS_BASE_PATH = "/work/dlclarge1/wagnerd-medquicktune/experiments/NePS" # "experiments/NePS"

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
    def setup_logging(log_file: Path | None = None) -> None:
        """Configure logging settings.
        
        Args:
            log_file: Optional path to log file. If provided, logs will be written to both
                     file and console. If None, logs only go to console.
        """
        # Check if logging is already configured
        root_logger = logging.getLogger()
        if root_logger.handlers:
            # Logging already configured, just add file handler if needed
            if log_file:
                # Check if file handler already exists for this file
                log_file_str = str(log_file.resolve())
                has_file_handler = any(
                    isinstance(h, logging.FileHandler) and h.baseFilename == log_file_str
                    for h in root_logger.handlers
                )
                if not has_file_handler:
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                    root_logger.addHandler(file_handler)
            return
        
        # First time setup
        handlers = [logging.StreamHandler()]
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers
        )

    def parse_neps_output(self) -> List[Dict[str, Any]]:
        """Parse the NePS summary CSV files from all outer folds into a list of dictionaries."""
        outer_fold_dirs = self._find_all_outer_folds()
        all_results = []
        
        # Track configs by their hyperparameters to avoid duplicates
        # Key: tuple of sorted config parameter values, Value: list of (config_dict, outer_fold_idx, config_id)
        config_map = {}
        
        for outer_fold_idx, outer_fold_dir in enumerate(outer_fold_dirs):
            summary_path = outer_fold_dir / SUMMARY_SUBDIR / SUMMARY_FILE
            if not summary_path.exists():
                # Fallback to old structure
                summary_path = self.neps_output / SUMMARY_SUBDIR / SUMMARY_FILE
                if not summary_path.exists():
                    continue
            
            df = pd.read_csv(summary_path)
            
            for _, row in df.iterrows():
                config_dict = self._extract_config_from_row(row, df.columns)
                config_id = int(row["id"])
                
                # Create a key from config parameters (excluding non-hyperparameter fields)
                excluded_keys = {"curves", "final_accuracy", "model_type", "dataset"}
                config_key = tuple(sorted([
                    (k, v) for k, v in config_dict.items() 
                    if k not in excluded_keys
                ]))
                
                if config_key not in config_map:
                    config_map[config_key] = []
                config_map[config_key].append((config_dict, outer_fold_idx, config_id))
        
        # For each unique configuration, aggregate across all outer folds
        for config_key, occurrences in config_map.items():
            # Use the first occurrence's config_dict as base
            base_config = occurrences[0][0]
            outer_fold_indices = [occ[1] for occ in occurrences]
            config_ids = [occ[2] for occ in occurrences]
            
            # Store outer fold info for later aggregation
            base_config["_outer_fold_indices"] = outer_fold_indices
            base_config["_config_ids"] = config_ids
            all_results.append(base_config)
        
        logging.info("Successfully parsed %d unique configurations from %d outer folds", 
                    len(all_results), len(outer_fold_dirs))
        return all_results
    
    def _find_summary_file(self) -> Path:
        """Find the NePS summary file, checking new structure first.
        Note: This method is kept for backward compatibility but parse_neps_output now uses all outer folds."""
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
            learning_curve_value = row["learning_curve"]
            # Handle nan values (pandas represents NaN as float('nan') or string 'nan')
            if pd.isna(learning_curve_value) or (isinstance(learning_curve_value, str) and learning_curve_value.lower() == 'nan'):
                config_dict["curves"] = []
            else:
                try:
                    config_dict["curves"] = ast.literal_eval(learning_curve_value)
                except (ValueError, SyntaxError) as e:
                    logging.warning(f"Could not parse learning_curve for config: {e}. Using empty list.")
                    config_dict["curves"] = []

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
        excluded_keys = {"curves", "final_accuracy", "model_type", "dataset", "_outer_fold_indices", "_config_ids"}
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
            "input_size_h": [meta_features["input_size_h"]] * len(results),
            "input_size_w": [meta_features["input_size_w"]] * len(results),
            "input_size_d": [meta_features["input_size_d"]] * len(results),
            "modality": [meta_features["modality"]] * len(results),
            "total_num_samples": [meta_features["total_num_samples"]] * len(results),
        }

        return pd.DataFrame(meta_data)
    
    def _extract_sample_counts_from_splits(self) -> Dict[str, int]:
        """Extract actual sample counts from CV splits files.
        
        Returns:
            Dictionary with "total_num_samples" if available
        """
        outer_fold_dirs = self._find_all_outer_folds()
        
        # Try to get total samples from outer CV splits (most reliable)
        dataset_name = self.hydra_config["data"]["dataset"]
        data_path = self.hydra_config.get("data", {}).get("path", "datasets")
        seed = self.hydra_config.get("seed", 42)
        cv_outer_folds_splits = self.hydra_config.get("cv_outer_folds_splits", 2)
        cv_outer_folds_repeats = self.hydra_config.get("cv_outer_folds_repeats", 1)
        
        total_num_samples = None
        
        # Try to find outer CV splits file
        from src.utils.common_utils import get_deterministic_cv_splits_path
        try:
            splits_dir, splits_file = get_deterministic_cv_splits_path(
                data_path, dataset_name, seed, cv_outer_folds_repeats, cv_outer_folds_splits, split_type="outer"
            )
            
            if Path(splits_file).exists():
                with open(splits_file, "rb") as f:
                    outer_splits_data = pickle.load(f)
                
                if "dataset_info" in outer_splits_data and "total_samples" in outer_splits_data["dataset_info"]:
                    total_num_samples = outer_splits_data["dataset_info"]["total_samples"]
        except Exception as e:
            logging.warning(f"Could not read outer CV splits: {e}")
        
        # Fallback: try to get from inner_cv_splits.pkl
        if total_num_samples is None:
            for outer_fold_dir in outer_fold_dirs:
                configs_dir = outer_fold_dir / CONFIGS_SUBDIR
                if configs_dir.exists():
                    config_dirs = [d for d in configs_dir.iterdir() if d.is_dir() and d.name.startswith(CONFIG_DIR_PREFIX)]
                    if config_dirs:
                        config_dir = config_dirs[0]
                        splits_file = config_dir / "inner_cv_splits.pkl"
                        
                        if splits_file.exists():
                            try:
                                with open(splits_file, "rb") as f:
                                    splits_data = pickle.load(f)
                                
                                # Get total samples from inner splits
                                if "total_samples" in splits_data:
                                    # This is train+val, but we'll use it as total if we can't get better info
                                    total_num_samples = splits_data["total_samples"]
                            except Exception as e:
                                logging.warning(f"Could not read splits file {splits_file}: {e}")
                        
                        break  # Only need to read from one config
        
        return {
            "total_num_samples": total_num_samples,
        }
    
    def _extract_spatial_size(self) -> Tuple[int, int, int]:
        """Extract input_size_h, input_size_w, input_size_d from spatial_size for 3D datasets or use default."""
        dataset = self.hydra_config["data"]["dataset"]
        dimensionality = self.hydra_config.get("data", {}).get("dimensionality", "3d").lower()
        
        # For 3D datasets, try to extract spatial_size from statistics
        if dimensionality == "3d":
            model_type = self.hydra_config.get("model", {}).get("type", "efficientnet")
            data_path = self.hydra_config.get("data", {}).get("path", "datasets")
            developer_mode = self.hydra_config.get("developer_mode", False)
            
            # Try to get voxel_calculation from first config (if available)
            outer_fold_dirs = self._find_all_outer_folds()
            voxel_calculation = "median"  # default
            for outer_fold_dir in outer_fold_dirs:
                configs_dir = outer_fold_dir / CONFIGS_SUBDIR
                if configs_dir.exists():
                    config_dirs = [d for d in configs_dir.iterdir() if d.is_dir() and d.name.startswith(CONFIG_DIR_PREFIX)]
                    if config_dirs:
                        config_file = config_dirs[0] / "config.yaml"
                        if config_file.exists():
                            try:
                                with open(config_file, "r") as f:
                                    config_data = yaml.safe_load(f)
                                    if "voxel_calculation" in config_data:
                                        voxel_calculation = config_data["voxel_calculation"]
                                        break
                            except Exception:
                                pass
            
            # Try to extract spatial_size using the same logic as in training
            try:
                from src.classification_3d.utils.dataset_info import extract_spatial_size
                spatial_size = extract_spatial_size(
                    model_type=model_type,
                    voxel_calculation=voxel_calculation,
                    dataset_name=dataset,
                    developer_mode=developer_mode,
                    data_path=data_path,
                    is_medmnist=False
                )
                
                if spatial_size is not None:
                    # For 3D, return (H, W, D) tuple
                    if isinstance(spatial_size, tuple) and len(spatial_size) == 3:
                        return (int(spatial_size[0]), int(spatial_size[1]), int(spatial_size[2]))
                    elif isinstance(spatial_size, (int, float)):
                        # If single value, use it for all dimensions
                        size = int(spatial_size)
                        return (size, size, size)
            except Exception as e:
                logging.warning(f"Could not extract spatial_size for {dataset}: {e}")
        
        # Fallback to default (for 2D or if extraction fails)
        default_h = DEFAULT_META_FEATURES["input_size_h"]
        default_w = DEFAULT_META_FEATURES["input_size_w"]
        default_d = DEFAULT_META_FEATURES["input_size_d"]
        return (default_h, default_w, default_d)
    
    def _get_modality(self, dataset_name: str) -> str:
        """Determine modality (CT or MRI) based on dataset name."""
        dataset_lower = dataset_name.lower()
        if dataset_lower in MRI_DATASETS:
            return "MRI"
        elif dataset_lower in CT_DATASETS:
            return "CT"
        else:
            # Default to CT if unknown, but log a warning
            logging.warning(f"Unknown dataset '{dataset_name}', defaulting to CT modality. Please add to MRI_DATASETS or CT_DATASETS if needed.")
            return DEFAULT_META_FEATURES["modality"]
    
    def _get_dataset_meta_features_from_quicktune_utils(self, dataset_name: str) -> Dict[str, Any] | None:
        """Get dataset-specific meta-features from custom_extract_image_dataset_metafeat.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with meta-features or None if extraction fails
        """
        try:
            # Create a temporary directory with the dataset name to satisfy the function's path requirement
            # The function uses path_root.name to get the dataset name, so we create a temp dir with that name
            with tempfile.TemporaryDirectory() as temp_base:
                temp_path = Path(temp_base) / dataset_name
                temp_path.mkdir(parents=True, exist_ok=True)
                
                # Call the function - it will use the directory name (dataset_name) to determine meta-features
                trial_info, metafeat = custom_extract_image_dataset_metafeat(temp_path)
                
                # Convert the metafeat to our format (matches quicktune_utils.py format)
                meta = {
                    "num_classes": metafeat.get("num_classes", DEFAULT_META_FEATURES["num_classes"]),
                    "input_channels": metafeat.get("input_channels", DEFAULT_META_FEATURES["input_channels"]),
                    "input_size_h": metafeat.get("input_size_h", DEFAULT_META_FEATURES["input_size_h"]),
                    "input_size_w": metafeat.get("input_size_w", DEFAULT_META_FEATURES["input_size_w"]),
                    "input_size_d": metafeat.get("input_size_d", DEFAULT_META_FEATURES["input_size_d"]),
                    "modality": metafeat.get("modality", DEFAULT_META_FEATURES["modality"]),
                    "total_num_samples": metafeat.get("total_num_samples", None),
                }
                return meta
        except Exception as e:
            logging.warning(f"Could not extract meta-features from quicktune_utils for dataset {dataset_name}: {e}")
            return None
    
    def _get_dataset_meta_features(self) -> Dict[str, Any]:
        """Get dataset-specific meta-features from config, CV splits, or return defaults."""
        dataset = self.hydra_config["data"]["dataset"]
        
        # First, try to get meta-features from quicktune_utils (dataset-specific defaults)
        quicktune_meta = self._get_dataset_meta_features_from_quicktune_utils(dataset)
        
        # Determine modality
        modality = self._get_modality(dataset)
        
        # Try to get meta-features from config first
        if "meta_features" in self.hydra_config.get("data", {}):
            config_meta = self.hydra_config["data"]["meta_features"]
            
            # Use quicktune defaults as fallback if not in config
            default_h = quicktune_meta["input_size_h"] if quicktune_meta else DEFAULT_META_FEATURES["input_size_h"]
            default_w = quicktune_meta["input_size_w"] if quicktune_meta else DEFAULT_META_FEATURES["input_size_w"]
            default_d = quicktune_meta["input_size_d"] if quicktune_meta else DEFAULT_META_FEATURES["input_size_d"]
            default_channels = quicktune_meta["input_channels"] if quicktune_meta else DEFAULT_META_FEATURES["input_channels"]
            default_classes = quicktune_meta["num_classes"] if quicktune_meta else DEFAULT_META_FEATURES["num_classes"]
            default_modality = quicktune_meta["modality"] if quicktune_meta else modality
            
            # Handle backward compatibility: if input_size exists, use it for all dimensions
            if "input_size" in config_meta and "input_size_h" not in config_meta:
                input_size = config_meta.get("input_size", default_h)
                input_size_h = input_size_w = input_size_d = input_size
            else:
                input_size_h = config_meta.get("input_size_h", default_h)
                input_size_w = config_meta.get("input_size_w", default_w)
                input_size_d = config_meta.get("input_size_d", default_d)
            
            # Get total_num_samples from config or use quicktune/default
            default_total_samples = quicktune_meta.get("total_num_samples") if quicktune_meta else DEFAULT_META_FEATURES["total_num_samples"]
            total_num_samples = config_meta.get("total_num_samples", 
                                                config_meta.get("total_train_samples", default_total_samples))  # Backward compatibility
            
            meta = {
                "num_classes": config_meta.get("num_classes", default_classes),
                "input_channels": config_meta.get("input_channels", default_channels),
                "input_size_h": input_size_h,
                "input_size_w": input_size_w,
                "input_size_d": input_size_d,
                "modality": config_meta.get("modality", default_modality),
                "total_num_samples": total_num_samples,
            }
        else:
            # Try to extract from CV splits
            split_counts = self._extract_sample_counts_from_splits()
            
            # Get total_num_samples from splits, quicktune_meta, or default
            total_num_samples = split_counts.get("total_num_samples")
            if total_num_samples is None:
                if quicktune_meta and quicktune_meta.get("total_num_samples"):
                    total_num_samples = quicktune_meta["total_num_samples"]
                else:
                    total_num_samples = DEFAULT_META_FEATURES["total_num_samples"]
            
            # Extract input_size_h, input_size_w, input_size_d from spatial_size for 3D datasets
            # Use quicktune defaults as fallback
            input_size_h, input_size_w, input_size_d = self._extract_spatial_size()
            if input_size_h == DEFAULT_META_FEATURES["input_size_h"] and quicktune_meta:
                # If extraction failed and we have quicktune defaults, use those
                input_size_h = quicktune_meta["input_size_h"]
                input_size_w = quicktune_meta["input_size_w"]
                input_size_d = quicktune_meta["input_size_d"]
            
            # Use quicktune defaults for other fields
            default_classes = quicktune_meta["num_classes"] if quicktune_meta else DEFAULT_META_FEATURES["num_classes"]
            default_channels = quicktune_meta["input_channels"] if quicktune_meta else DEFAULT_META_FEATURES["input_channels"]
            default_modality = quicktune_meta["modality"] if quicktune_meta else modality
            
            meta = {
                "num_classes": default_classes,
                "input_channels": default_channels,
                "input_size_h": input_size_h,
                "input_size_w": input_size_w,
                "input_size_d": input_size_d,
                "modality": default_modality,
                "total_num_samples": total_num_samples,
            }
        
        return meta
    
    def _find_all_outer_folds(self) -> List[Path]:
        """Find all cv_outer_fold directories."""
        outer_fold_dirs = sorted(
            [d for d in self.neps_output.iterdir() 
             if d.is_dir() and d.name.startswith("cv_outer_fold_")],
            key=lambda x: int(x.name.split("_")[-1])
        )
        return outer_fold_dirs if outer_fold_dirs else [self.neps_output / CV_FOLD_DIR]
    
    def _create_cost_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create cost DataFrame by reading evaluation_duration from report.yaml files.
        Aggregates costs across all outer folds by averaging."""
        costs = []
        outer_fold_dirs = self._find_all_outer_folds()
        
        for idx, result in enumerate(results, start=1):
            fold_costs = []
            
            # Get outer fold indices and config IDs for this result
            outer_fold_indices = result.get("_outer_fold_indices", list(range(len(outer_fold_dirs))))
            config_ids = result.get("_config_ids", [idx] * len(outer_fold_indices))
            
            # Collect costs from all outer folds where this config exists
            for outer_fold_idx, config_id in zip(outer_fold_indices, config_ids):
                if outer_fold_idx < len(outer_fold_dirs):
                    outer_fold_dir = outer_fold_dirs[outer_fold_idx]
                    config_dir = outer_fold_dir / CONFIGS_SUBDIR / f"{CONFIG_DIR_PREFIX}{config_id}"
                    report_path = config_dir / "report.yaml"
                    
                    if report_path.exists():
                        with open(report_path, "r", encoding="utf-8") as f:
                            report_data = yaml.safe_load(f)
                            # Use evaluation_duration as cost
                            cost = report_data.get("evaluation_duration", 1.0)
                            fold_costs.append(cost)
            
            # Also check old structure as fallback
            if not fold_costs:
                old_config_dir = self.neps_output / CONFIGS_SUBDIR / f"{CONFIG_DIR_PREFIX}{idx}_0"
                old_report_path = old_config_dir / "report.yaml"
                if old_report_path.exists():
                    with open(old_report_path, "r", encoding="utf-8") as f:
                        report_data = yaml.safe_load(f)
                        cost = report_data.get("evaluation_duration", 1.0)
                        fold_costs.append(cost)
            
            if fold_costs:
                # Average cost across all outer folds
                avg_cost = np.mean(fold_costs)
                costs.append(avg_cost)
            else:
                logging.warning(f"Report file not found for config {idx} in any outer fold, using default cost of 1")
                costs.append(1.0)
        
        cost_data = {
            "cost": costs
        }
        
        return pd.DataFrame(cost_data)
    
    def _create_curves_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create learning curves DataFrame.
        Aggregates curves across all outer folds and inner folds."""
        curves_data = []
        outer_fold_dirs = self._find_all_outer_folds()

        for idx, result in enumerate(results, start=1):  # Start enumeration at 1
            all_fold_curves = []
            
            # Get outer fold indices and config IDs for this result
            outer_fold_indices = result.get("_outer_fold_indices", list(range(len(outer_fold_dirs))))
            config_ids = result.get("_config_ids", [idx] * len(outer_fold_indices))
            
            # Collect curves from all outer folds where this config exists
            for outer_fold_idx, config_id in zip(outer_fold_indices, config_ids):
                if outer_fold_idx < len(outer_fold_dirs):
                    outer_fold_dir = outer_fold_dirs[outer_fold_idx]
                    config_dir = outer_fold_dir / CONFIGS_SUBDIR / f"{CONFIG_DIR_PREFIX}{config_id}"
                    
                    if config_dir.exists():
                        cv_inner_folds = self._count_folds(config_dir)
                        fold_curves = self.get_fold_metrics(config_id, cv_inner_folds, str(config_dir))
                        
                        if fold_curves:
                            all_fold_curves.extend(fold_curves)
            
            # Also check old structure as fallback
            if not all_fold_curves:
                old_config_dir = self.neps_output / CONFIGS_SUBDIR / f"{CONFIG_DIR_PREFIX}{idx}_0"
                if old_config_dir.exists():
                    cv_inner_folds = self._count_folds(old_config_dir)
                    fold_curves = self.get_fold_metrics(idx, cv_inner_folds, str(old_config_dir))
                    if fold_curves:
                        all_fold_curves.extend(fold_curves)

            if all_fold_curves:
                # Ensure all folds have same number of epochs
                min_epochs = min(len(curve) for curve in all_fold_curves)
                all_fold_curves = [curve[:min_epochs] for curve in all_fold_curves]

                # Average over all folds (outer and inner) for each epoch
                all_fold_curves = np.array(all_fold_curves)
                avg_curve = np.mean(all_fold_curves, axis=0)
                curves_data.append(avg_curve)
            else:
                logging.warning(f"No valid curves found for config {idx}")
                curves_data.append(np.zeros(1))

        return pd.DataFrame(curves_data)
    
    def _find_config_directory(self, config_idx: int) -> Path:
        """Find the configuration directory, checking new structure first.
        Returns the first available config directory (for backward compatibility)."""
        # Check for new outer fold structure first, then fall back to old structure
        outer_fold_dirs = self._find_all_outer_folds()
        if outer_fold_dirs:
            config_dir = outer_fold_dirs[0] / CONFIGS_SUBDIR / f"{CONFIG_DIR_PREFIX}{config_idx}"
            if config_dir.exists():
                return config_dir
        # Fallback to old structure
        config_dir = self.neps_output / CONFIGS_SUBDIR / f"{CONFIG_DIR_PREFIX}{config_idx}_0"
        return config_dir
    
    def _count_folds(self, config_dir: Path) -> int:
        """Count the number of fold directories in a config directory."""
        return sum(1 for d in config_dir.iterdir() 
                  if d.is_dir() and d.name.startswith(FOLD_DIR_PREFIX))

    def get_fold_metrics(self, config_idx: int, cv_inner_folds: int, config_dir: str = None) -> List[np.ndarray]:
        """Helper function to read metrics from each fold."""
        fold_curves = []
        
        # Use provided config_dir or construct it
        if config_dir is None:
            config_dir = str(self._find_config_directory(config_idx))

        for fold in range(cv_inner_folds):
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
    experiments_base_path: str | Path | None = None,
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
    # Create portfolio directory and set up logging to file
    portfolio_dir = Path(output_dir)
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = portfolio_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "create_portfolio_cluster.log"
    PortfolioCreator.setup_logging(log_file=log_file)
    
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
    
    # Use provided base path, environment variable, or default
    if experiments_base_path is None:
        experiments_base_path = EXPERIMENTS_BASE_PATH
    experiments_base = Path(experiments_base_path)
    logging.info(f"Using experiments base path: {experiments_base}")

    # Process each dataset with its specific experiments
    for dataset, experiment_spec in dataset_experiment_pairs:
        # base_path = Path(EXPERIMENTS_BASE_PATH) / dataset
        base_path = experiments_base / dataset

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

    # Get experiments base path from config or use default
    experiments_base_path = config.get("experiments_base_path", None)

    # Convert portfolio_dir to absolute path to avoid issues with relative paths
    portfolio_dir = Path(config.portfolio_dir).resolve()
        
    merge_neps_runs_multi_dataset(
        dataset_spec=dataset_spec,
        output_dir=portfolio_dir,
        experiments_base_path=experiments_base_path
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
