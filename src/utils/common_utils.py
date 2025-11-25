import os
import random
import hashlib
import warnings

import neps
import numpy as np
import torch
import yaml
import shutil


def get_cache_file_path(data_path, dataset_name, dimensionality, cv_outer_fold, voxel_calculation=None):
    """
    Generate cache file path for different dataset types and configurations.
    
    Args:
        data_path (Path): Base path to datasets
        dataset_name (str): Name of the dataset
        dimensionality (str): '2d' or '3d'
        cv_outer_fold (int): Cross-validation fold number
        voxel_calculation (str, optional): Voxel calculation method for 3D
        
    Returns:
        Path: Cache file path
    """
    if dimensionality == "2d":
        cache_file = (
            data_path
            / "cache"
            / f"{dataset_name}_2d_cv{cv_outer_fold}.pkl"
        )
    elif dimensionality == "3d":
        cache_file = (
            data_path
            / "cache"
            / f"{dataset_name}_3d_cv{cv_outer_fold}_voxel{voxel_calculation}.pkl"
        )
    else:
        raise ValueError(f"Unsupported dimensionality: {dimensionality}")
    
    return cache_file


def get_deterministic_cv_splits_path(data_path, dataset_name, seed, cv_folds_repeats, cv_folds_splits, split_type="outer"):
    """
    Generate a deterministic path for CV splits that is independent of experiment name.
    Uses MD5 hash of dataset name, seed, and CV parameters to ensure reproducibility.
    
    Args:
        data_path (str): Base path to datasets directory
        dataset_name (str): Name of the dataset
        seed (int): Random seed used for CV splits
        cv_folds_repeats (int): Number of repeats for repeated stratified K-fold
        cv_folds_splits (int): Number of splits per repeat
        split_type (str): Type of split - "outer" or "inner" (default: "outer")
        
    Returns:
        tuple: (splits_directory, splits_file_path) as strings
    """
    # Create a deterministic hash from parameters that define the CV splits
    # This ensures the same splits are used regardless of experiment name
    hash_input = f"{dataset_name}_{seed}_{cv_folds_repeats}_{cv_folds_splits}_{split_type}"
    hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:16]
    
    # Store splits in datasets/cache/cv_splits/ to be independent of experiment directories
    splits_dir = os.path.join(data_path, "cache", "cv_splits")
    splits_file = os.path.join(
        splits_dir,
        f"cv_splits_{split_type}_{dataset_name}_seed{seed}_r{cv_folds_repeats}_s{cv_folds_splits}_{hash_value}.pkl"
    )
    
    return splits_dir, splits_file


def get_deterministic_inner_cv_splits_path(data_path, dataset_name, seed, cv_inner_folds_repeats, cv_inner_folds_splits, data_hash=None):
    """
    Generate a deterministic path for inner CV splits based on the train_val data.
    Uses a hash of the data identifiers to ensure splits are consistent for the same data.
    
    Args:
        data_path (str): Base path to datasets directory
        dataset_name (str): Name of the dataset
        seed (int): Random seed used for CV splits
        cv_inner_folds_repeats (int): Number of repeats for repeated stratified K-fold
        cv_inner_folds_splits (int): Number of splits per repeat
        data_hash (str, optional): Hash of the data identifiers (e.g., sorted file paths).
                                   If None, will use a hash based on dataset and seed.
        
    Returns:
        tuple: (splits_directory, splits_file_path) as strings
    """
    if data_hash is None:
        # Fallback: use dataset name and seed if data hash is not provided
        # This is less ideal but ensures determinism
        hash_input = f"{dataset_name}_{seed}_{cv_inner_folds_repeats}_{cv_inner_folds_splits}_inner"
    else:
        # Use hash of data identifiers to ensure splits are consistent for the same data
        hash_input = f"{dataset_name}_{seed}_{cv_inner_folds_repeats}_{cv_inner_folds_splits}_{data_hash}_inner"
    
    hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:16]
    
    # Store splits in datasets/cache/cv_splits/ to be independent of experiment directories
    splits_dir = os.path.join(data_path, "cache", "cv_splits")
    splits_file = os.path.join(
        splits_dir,
        f"cv_splits_inner_{dataset_name}_seed{seed}_r{cv_inner_folds_repeats}_s{cv_inner_folds_splits}_{hash_value}.pkl"
    )
    
    return splits_dir, splits_file

    
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
        param_kwargs = {}
        for k, v in param_config.items():
            if k not in ["type", "is_fidelity"]:
                # Handle scientific notation strings (e.g., "1e-6")
                if isinstance(v, str) and ("e" in v.lower()):
                    try:
                        param_kwargs[k] = float(v)
                    except ValueError:
                        # Keep as string if conversion fails (e.g., "efficientnet-b0")
                        param_kwargs[k] = v
                else:
                    param_kwargs[k] = v

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
    
    # Filter warnings for non-deterministic 3D pooling operations
    # These operations (avg_pool3d_backward, max_pool3d_with_indices_backward) don't have
    # deterministic CUDA implementations yet, but this doesn't significantly affect reproducibility
    # since most other operations are deterministic. The warnings are noisy but harmless.
    warnings.filterwarnings(
        "ignore",
        message=".*does not have a deterministic implementation.*",
        category=UserWarning,
        module="torch.autograd.graph"
    )
    
    # Use deterministic algorithms where available (PyTorch 1.8+)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # Fallback for older PyTorch versions
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed

def cleanup_training_artifacts(pipeline_directory, total_inner_folds):
    """
    Delete model checkpoints after test evaluation to save disk space.
    
    Args:
        pipeline_directory (str): Directory containing the pipeline results
        total_inner_folds (int): Total number of cross-validation folds (repeats * splits)
    """
    print(f"\n{'='*80}")
    print(f"CLEANING UP TRAINING ARTIFACTS")
    print(f"{'='*80}")
    
    total_deleted_size = 0
    deleted_files = 0
    
    # Delete checkpoints for each fold
    for fold in range(total_inner_folds):
        fold_dir = os.path.join(pipeline_directory, f"cv_inner_fold_{fold}")
        
        if os.path.exists(fold_dir):
            # Delete model checkpoint files
            checkpoint_files = [
                "model_latest_checkpoint.pth",
                "best_model_checkpoint.pth"
            ]
            
            for checkpoint_file in checkpoint_files:
                checkpoint_path = os.path.join(fold_dir, checkpoint_file)
                if os.path.exists(checkpoint_path):
                    try:
                        file_size = os.path.getsize(checkpoint_path)
                        os.remove(checkpoint_path)
                        total_deleted_size += file_size
                        deleted_files += 1
                        print(f"Deleted: {checkpoint_path} ({file_size / (1024*1024):.2f} MB)")
                    except OSError as e:
                        print(f"Warning: Could not delete {checkpoint_path}: {e}")
            
            # Delete entire fold directory if it's empty (except for normalization_stats.json)
            try:
                remaining_files = [f for f in os.listdir(fold_dir) if f != "normalization_stats.json"]
                if not remaining_files:
                    shutil.rmtree(fold_dir)
                    print(f"Deleted empty fold directory: {fold_dir}")
            except OSError as e:
                print(f"Warning: Could not delete fold directory {fold_dir}: {e}")
    
    # Summary
    print(f"\nCleanup Summary:")
    print(f"  Files deleted: {deleted_files}")
    print(f"  Total space freed: {total_deleted_size / (1024*1024):.2f} MB")
    print(f"{'='*80}\n")
