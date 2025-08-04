import os
import random

import neps
import numpy as np
import torch
import yaml


def get_cache_file_path(data_path, dataset_name, dimensionality, cv_fold, voxel_calculation=None):
    """
    Generate cache file path for different dataset types and configurations.
    
    Args:
        data_path (Path): Base path to datasets
        dataset_name (str): Name of the dataset
        dimensionality (str): '2d' or '3d'
        cv_fold (int): Cross-validation fold number
        voxel_calculation (str, optional): Voxel calculation method for 3D
        
    Returns:
        Path: Cache file path
    """
    if dimensionality == "2d":
        cache_file = (
            data_path
            / "cache"
            / f"{dataset_name}_2d_cv{cv_fold}.pkl"
        )
    elif dimensionality == "3d":
        cache_file = (
            data_path
            / "cache"
            / f"{dataset_name}_3d_cv{cv_fold}_voxel{voxel_calculation}.pkl"
        )
    else:
        raise ValueError(f"Unsupported dimensionality: {dimensionality}")
    
    return cache_file

    
def yaml_to_neps_pipeline_space(yaml_path):
    """Convert YAML pipeline space configuration to NePS format.

    Args:
        yaml_path (str): Path to YAML configuration file

    Returns:
        dict: NePS-compatible pipeline space configuration
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline_space = {}

    for key, param_config in config.items():
        param_kwargs = {
            k: float(v) if isinstance(v, str) and ("e" in v.lower()) else v
            for k, v in param_config.items()
            if k not in ["type", "is_fidelity"]
        }

        # Create parameter
        if param_config["type"] == "float":
            param = neps.Float(**param_kwargs)
        elif param_config["type"] == "int":
            param = neps.Integer(**param_kwargs)
        elif param_config["type"] == "categorical":
            param = neps.Categorical(**param_kwargs)

        # Add is_fidelity if specified
        if param_config.get("is_fidelity", False):
            param.is_fidelity = True

        pipeline_space[key] = param

    return pipeline_space


def neps_space_to_dict(pipeline_space):
    """Convert NePS pipeline space to a dictionary format suitable for YAML serialization."""
    space_dict = {}
    for key, value in pipeline_space.items():
        param_dict = {
            "type": value.__class__.__name__.lower(),
            "lower": value.lower if hasattr(value, "lower") else None,
            "upper": value.upper if hasattr(value, "upper") else None,
            "log": value.log if hasattr(value, "log") else None,
            "choices": value.choices if hasattr(value, "choices") else None,
            "is_fidelity": (
                value.is_fidelity if hasattr(value, "is_fidelity") else False
            ),
        }
        # Remove None values
        space_dict[key] = {k: v for k, v in param_dict.items() if v is not None}
    return space_dict


def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed (int): Random seed value

    Returns:
        None
    """
    random.seed(seed)  # Python's random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed
