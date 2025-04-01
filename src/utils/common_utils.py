import os
import random

import neps
import numpy as np
import torch
import yaml


def yaml_to_neps_pipeline_space(yaml_path):
    """
    Parse YAML configuration file and convert to NePS pipeline space format.
    Supports both configurations with and without user priors.

    Args:
        yaml_path (str): Path to the YAML configuration file

    Returns:
        dict: NePS-compatible pipeline space dictionary

    Raises:
        ValueError: If unknown parameter type is encountered
    """
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline_space = {}

    # Check if we're using priorband (with user priors) or not
    using_priorband = any(
        "default" in param and "default_confidence" in param
        for param in config.values()
        if isinstance(param, dict)
    )

    for key, value in config.items():
        # Skip non-hyperparameter entries
        if not isinstance(value, dict) or "type" not in value:
            print(f"Skipping non-hyperparameter '{key}': {value}")
            continue

        param_type = value.get("type")
        is_fidelity = value.get("is_fidelity", False)

        # Base parameters for all types
        param_kwargs = {}
        if "lower" in value:
            param_kwargs["lower"] = value["lower"]
        if "upper" in value:
            param_kwargs["upper"] = value["upper"]

        # Handle user priors if present and using priorband
        if using_priorband and "default" in value and "default_confidence" in value:
            param_kwargs.update(
                {
                    "default": value["default"],
                    "default_confidence": value["default_confidence"],
                }
            )

        # Parameter-specific configuration
        if param_type == "float":
            param_kwargs["log"] = value.get("log", False)
            pipeline_space[key] = neps.Float(**param_kwargs)
        elif param_type == "int":
            if is_fidelity:
                param_kwargs["is_fidelity"] = True
            pipeline_space[key] = neps.Integer(**param_kwargs)
        elif param_type == "categorical":
            param_kwargs.pop("lower", None)
            param_kwargs.pop("upper", None)
            param_kwargs["choices"] = value.get("choices")
            pipeline_space[key] = neps.Categorical(**param_kwargs)
        else:
            raise ValueError(f"Unknown type '{param_type}' for parameter '{key}'")

    # Log the configuration mode
    print(f"Configuration mode: {'with' if using_priorband else 'without'} user priors")

    return pipeline_space


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