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
    