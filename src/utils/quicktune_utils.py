import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms  # type: ignore
from torchvision.datasets import ImageFolder  # type: ignore
from omegaconf import DictConfig  # Add this import at the top
from ConfigSpace import ConfigurationSpace
from qtt.predictors import PerfPredictor, CostPredictor

from ConfigSpace import (CategoricalHyperparameter, ConfigurationSpace,
                         UniformFloatHyperparameter,
                         UniformIntegerHyperparameter)
from dataclasses import dataclass


def custom_extract_image_dataset_metafeat(
    path_root: str | Path, train_split: str = "train", val_split: str = "val"
):
    """
    Extracts metadata features from an image dataset for classification tasks.
    Automatically detects if the dataset is pre-split (train/val) or unsplit.

    This function analyzes the specified dataset directory to compute metadata
    features, such as the number of samples, number of classes, average number
    of features (image size), and number of channels.

    Args:
        path_root (str | Path): The root directory of the dataset.
        train_split (str, optional): The subdirectory name for training data. Defaults to "train".
        val_split (str, optional): The subdirectory name for validation data. Defaults to "val".

    Returns:
        tuple: A tuple containing:
            - trial_info (dict): Information about the dataset directory and splits.
            - metafeat (dict): Metadata features including:
                - "num-samples": Total number of samples in the dataset.
                - "num-classes": Number of classes in the dataset.
                - "num-features": Average number of features (image size).
                - "num-channels": Number of channels in the images.

    Raises:
        ValueError: If the specified path does not exist or is not a directory.
    """
    # handle path
    path_root = Path(path_root)
    path_root = path_root.expanduser()  # expands ~ to home directory
    path_root = path_root.resolve()  # convert to an absolute path
    if not path_root.exists():
        raise ValueError(f"The specified path does not exist: {path_root}")
    if not path_root.is_dir():
        raise ValueError(f"The specified path is not a directory: {path_root}")

    num_samples = 0
    num_classes = 0
    num_features = 224
    num_channels = 3

    # Check if dataset is pre-split
    train_path = path_root / train_split
    val_path = path_root / val_split
    is_presplit = train_path.exists() or val_path.exists()

    if is_presplit:
        # QuickTune's default extract_image_dataset_metafeat() implementation
        # Handle pre-split dataset (existing logic)
        if train_path.exists():
            trainset = ImageFolder(train_path)
            num_samples += len(trainset)
            num_channels = 3 if trainset[0][0].mode == "RGB" else 1
            num_classes = len(trainset.classes)

            for img, _ in trainset:
                num_features += img.size[0]
            num_features //= len(trainset)

        if os.path.exists(val_path):
            valset = ImageFolder(val_path)
            num_samples += len(valset)
    
    else:
        # Handle unsplit dataset (single directory)
        try:
            # TODO: fix hardcoding for brain_tumor dataset > calculated in load_brain_tumor_dataset()
            num_samples = 253
            num_classes = 2
            num_features = 224
            num_channels = 3

        except Exception as e:
            raise ValueError(f"Could not process dataset directory: {str(e)}")

    # Output as needed for QuickTune
    # TODO: for medical datasets might be good to add class distribution?: brain_tumor: {0: 98, 1: 155}
    metafeat = {
        "num-samples": num_samples,
        "num-classes": num_classes,
        "num-features": num_features,
        "num-channels": num_channels,
    }

    trial_info = {
        "data-dir": str(path_root),
        "train-split": train_split,
        "val-split": val_split,
        "num-classes": num_classes,
    }

    return trial_info, metafeat


class CustomCostPredictor(CostPredictor):
    """Custom CostPredictor with modified default parameters"""
    
    def __init__(self, **kwargs):
        # Override default parameters
        # Batch size needs to be reducedto avoid division by zero for small datasets
        custom_defaults = {
            "learning_rate_init": 0.0001,
            "batch_size": 1,  # default: 1024
            "max_iter": 100,
            "early_stop": True,
            "patience": 5,
            "validation_fraction": 0.1,
            "tol": 1e-4,
        }
        super().__init__(fit_params=custom_defaults, **kwargs)


class CustomPerfPredictor(PerfPredictor):
    """Custom PerfPredictor with modified default parameters"""
    
    def __init__(self, **kwargs):
        # Override default parameters
        custom_defaults = {
            "learning_rate_init": 0.0001,
            "batch_size": 1,  # default: 1024
            "max_iter": 100,
            "early_stop": True,
            "patience": 5,
            "validation_fraction": 0.1,
            "tol": 1e-4,
        }
        super().__init__(fit_params=custom_defaults, **kwargs)

@dataclass
class PortfolioData:
    """Container for portfolio data files"""

    pipeline_df: pd.DataFrame
    curve_df: pd.DataFrame
    cost_df: pd.DataFrame
    meta_df: pd.DataFrame


class ConfigSpaceBuilder:
    """Handles creation of ConfigurationSpace from YAML"""

    @staticmethod
    def from_yaml(config_dict: dict) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        type_to_param = {
            "float": UniformFloatHyperparameter,
            "int": UniformIntegerHyperparameter,
            "categorical": CategoricalHyperparameter,
        }

        for param_name, param_config in config_dict.items():
            param_type = param_config["type"]
            param_class = type_to_param.get(param_type)

            if not param_class:
                raise ValueError(f"Unknown parameter type: {param_type}")

            # Convert scientific notation strings to float
            if param_type in ["float", "int"]:
                lower = (
                    float(param_config["lower"])
                    if isinstance(param_config["lower"], str)
                    else param_config["lower"]
                )
                upper = (
                    float(param_config["upper"])
                    if isinstance(param_config["upper"], str)
                    else param_config["upper"]
                )

                if param_type == "float":
                    param = param_class(
                        name=param_name,
                        lower=lower,
                        upper=upper,
                        log=param_config.get("log", False),
                    )
                else:  # int
                    param = param_class(
                        name=param_name, lower=int(lower), upper=int(upper)
                    )
            else:  # categorical
                param = param_class(name=param_name, choices=param_config["choices"])

            cs.add_hyperparameter(param)

        return cs


class PortfolioManager:
    """Handles loading and saving of portfolio data"""

    @staticmethod
    def load(portfolio_dir: str) -> PortfolioData:
        """Load portfolio data from CSV files"""
        try:
            return PortfolioData(
                pipeline_df=pd.read_csv(f"{portfolio_dir}/config.csv", index_col=0),
                curve_df=pd.read_csv(f"{portfolio_dir}/curve.csv", index_col=0),
                cost_df=pd.read_csv(f"{portfolio_dir}/cost.csv", index_col=0),
                meta_df=pd.read_csv(f"{portfolio_dir}/meta.csv", index_col=0),
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Portfolio file not found in {portfolio_dir}: {e}")

    @staticmethod
    def save(portfolio: PortfolioData, output_dir: str):
        """Save portfolio data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        portfolio.pipeline_df.to_csv(f"{output_dir}/config.csv", index=True)
        portfolio.curve_df.to_csv(f"{output_dir}/curve.csv", index=True)
        portfolio.cost_df.to_csv(f"{output_dir}/cost.csv", index=True)
        portfolio.meta_df.to_csv(f"{output_dir}/meta.csv", index=True)
    