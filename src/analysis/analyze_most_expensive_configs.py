#!/usr/bin/env python3
"""
Analyze which dataset-model-voxel_calculation combinations are most expensive for 50 epochs.

This script:
1. Iterates through all datasets, models, search spaces, and voxel_calculations
2. Extracts spatial_size for each combination (using maximum dimensions)
3. Estimates model parameters and memory usage
4. Calculates cost for 50 epochs
5. Ranks combinations by cost
"""

import os
import sys
import re
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.classification_3d.utils.dataset_info import extract_spatial_size
from src.classification_3d.models_3d import get_3d_model


# Configuration
DATASETS = ["gist", "lipo", "melanoma", "liver", "desmoid", "crlm"]
MODELS = ["densenet", "resnet", "efficientnet", "swin_unetr", "vit"]
SEARCH_SPACES = {
    "densenet": "configs/pipeline_spaces/densenet.yaml",
    "resnet": "configs/pipeline_spaces/resnet.yaml",
    "efficientnet": "configs/pipeline_spaces/efficientnet.yaml",
    "swin_unetr": "configs/pipeline_spaces/swinunetr.yaml",
    "vit": "configs/pipeline_spaces/vit.yaml",
}
TRAINING_SPACE = "configs/pipeline_spaces/training.yaml"
EXPERIMENTAL_SETTING = "configs/experimental_setting.yaml"
VOXEL_CALCULATIONS = ["mean", "median", "isotropic", "volumetric_isotropic"]
DATA_PATH = "datasets"
NUM_CLASSES = 2  # Binary classification
EPOCHS = 50


def load_search_space(yaml_path: str) -> Dict:
    """Load search space from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_cv_settings(
    yaml_path: str,
    cv_outer_folds_repeats: Optional[int] = None,
    cv_outer_folds_splits: Optional[int] = None,
    cv_inner_folds: Optional[int] = None
) -> Dict:
    """
    Load CV settings from experimental_setting.yaml or use provided parameters.
    
    Args:
        yaml_path: Path to experimental_setting.yaml
        cv_outer_folds_repeats: Override cv_outer_folds_repeats (optional)
        cv_outer_folds_splits: Override cv_outer_folds_splits (optional)
        cv_inner_folds: Override cv_inner_folds (optional)
    """
    # Load from YAML if parameters not provided
    if any(x is None for x in [cv_outer_folds_repeats, cv_outer_folds_splits, cv_inner_folds]):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if cv_outer_folds_repeats is None:
            cv_outer_folds_repeats = config.get("cv_outer_folds_repeats", 5)
        if cv_outer_folds_splits is None:
            cv_outer_folds_splits = config.get("cv_outer_folds_splits", 3)
        if cv_inner_folds is None:
            cv_inner_folds = config.get("cv_inner_folds", 5)
    
    # Total outer folds = repeats * splits
    total_outer_folds = cv_outer_folds_repeats * cv_outer_folds_splits
    
    return {
        "cv_outer_folds_repeats": cv_outer_folds_repeats,
        "cv_outer_folds_splits": cv_outer_folds_splits,
        "cv_inner_folds": cv_inner_folds,
        "total_outer_folds": total_outer_folds,
        "total_trainings_per_config": total_outer_folds * cv_inner_folds
    }


def get_max_hyperparameters(search_space: Dict, model_type: str = None) -> Dict:
    """
    Extract maximum/expensive hyperparameters from search space.
    For categorical, use the last choice (usually largest).
    For int/float, use upper bound.
    
    Special case: For patch_size and patch_size_0, use MINIMUM values
    because smaller patch sizes = more tokens = more VRAM (especially for transformers).
    """
    hyperparameters = {}
    
    # Parameters where smaller values are more expensive (more VRAM)
    # These are typically patch sizes that affect token count
    expensive_small_params = {"patch_size", "patch_size_0"}
    
    for key, value in search_space.items():
        if not isinstance(value, dict):
            continue
            
        param_type = value.get("type")
        
        # Special handling for patch_size parameters (smaller = more expensive)
        if key in expensive_small_params:
            if param_type == "categorical":
                choices = value.get("choices", [])
                if choices and isinstance(choices[0], (int, float)):
                    hyperparameters[key] = min(choices)  # Smallest = most expensive
                else:
                    hyperparameters[key] = choices[0]  # First = smallest
            elif param_type == "int":
                hyperparameters[key] = value.get("lower", 1)  # Lower bound = smallest
            continue
        
        # Normal handling for other parameters
        if param_type == "categorical":
            # Use last choice (often largest)
            choices = value.get("choices", [])
            if choices:
                # For strings, use last; for numbers, find max
                if isinstance(choices[0], (int, float)):
                    hyperparameters[key] = max(choices)
                else:
                    hyperparameters[key] = choices[-1]
        elif param_type == "int":
            hyperparameters[key] = value.get("upper", value.get("lower", 1))
        elif param_type == "float":
            hyperparameters[key] = value.get("upper", value.get("lower", 0.0))
        elif param_type == "bool":
            hyperparameters[key] = True
    
    return hyperparameters


def estimate_model_parameters(model_type: str, hyperparameters: Dict, spatial_size: Optional[Tuple], num_classes: int, is_medmnist: bool = False) -> int:
    """
    Estimate model parameters by instantiating the model.
    Returns number of trainable parameters.
    """
    try:
        model_config = {
            "type": model_type,
            "task": "classification",
            "num_classes": num_classes
        }
        
        # Create model
        if model_type in ["vit", "swin_unetr"]:
            if spatial_size is None:
                return 0
            model = get_3d_model(
                model_config=model_config,
                hyperparameters=hyperparameters,
                developer_mode=False,
                spatial_size=spatial_size,
                is_medmnist=is_medmnist
            )
        else:
            model = get_3d_model(
                model_config=model_config,
                hyperparameters=hyperparameters,
                developer_mode=False,
                spatial_size=None,
                is_medmnist=is_medmnist
            )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return num_params
    except Exception as e:
        print(f"  Warning: Could not estimate parameters for {model_type}: {e}")
        return 0


def estimate_memory_per_iteration(
    model_type: str,
    num_params: int,
    spatial_size: Optional[Tuple],
    batch_size: int = 1
) -> float:
    """
    Estimate memory usage per iteration in GB.
    Rough approximation based on:
    - Model parameters: 4 bytes per float32 parameter
    - Activations: depends on spatial_size and model architecture
    - Gradients: same as parameters
    """
    # Model parameters memory (GB)
    model_memory = (num_params * 4) / (1024 ** 3)  # 4 bytes per float32
    
    # Gradient memory (same as parameters)
    gradient_memory = model_memory
    
    # Activation memory (rough estimate)
    if spatial_size is None:
        # For CNNs without spatial_size, use conservative estimate
        activation_memory = 0.5  # Conservative estimate for CNNs
    else:
        # For transformers and CNNs with spatial_size, estimate based on spatial_size
        h, w, d = spatial_size
        
        if model_type == "vit":
            # ViT: tokens = (h * w * d) / (patch_size^3)
            # Rough estimate: assume patch_size=16
            tokens = (h * w * d) / (16 ** 3)
            # Hidden size estimate (from hyperparameters, use max)
            hidden_size = 12 * 64  # max hidden_size_multiplier * 12
            # Activation memory: tokens * hidden_size * num_layers * 4 bytes
            num_layers = 12  # max
            activation_memory = (tokens * hidden_size * num_layers * 4) / (1024 ** 3)
        elif model_type == "swin_unetr":
            # SwinUNETR: more complex, rough estimate
            tokens = (h * w * d) / (4 ** 3)  # patch_size=4
            feature_size = 48  # max
            num_stages = 4
            activation_memory = (tokens * feature_size * num_stages * 4) / (1024 ** 3)
        else:
            # For CNNs (DenseNet, ResNet, EfficientNet) with spatial_size
            # Estimate based on feature map sizes through the network
            # Typical CNN: input -> conv layers -> pooling -> ... -> final features
            # We estimate memory for intermediate feature maps
            
            # Input volume
            input_volume = h * w * d * batch_size  # voxels
            
            # Estimate feature map sizes through network
            # After first conv+pool: roughly (h/2, w/2, d/2) with ~64 channels
            # After second stage: roughly (h/4, w/4, d/4) with ~128 channels
            # After third stage: roughly (h/8, w/8, d/8) with ~256 channels
            # After fourth stage: roughly (h/16, w/16, d/16) with ~512 channels
            
            # Rough estimate: sum of feature maps at different stages
            # We use a simplified model: assume average feature map size
            # across all stages is roughly (h/4, w/4, d/4) with average ~256 channels
            avg_feature_h = max(1, h // 4)
            avg_feature_w = max(1, w // 4)
            avg_feature_d = max(1, d // 4)
            avg_channels = 256  # Conservative estimate for average channels
            
            # Feature map memory per stage (rough estimate)
            feature_map_volume = avg_feature_h * avg_feature_w * avg_feature_d * avg_channels * batch_size
            # Assume we store activations for ~4-5 stages simultaneously during forward+backward
            num_stages_active = 4
            activation_memory = (feature_map_volume * num_stages_active * 4) / (1024 ** 3)  # 4 bytes per float32
            
            # Add input memory
            input_memory = (input_volume * 4) / (1024 ** 3)
            activation_memory += input_memory
            
            # Ensure minimum reasonable value
            activation_memory = max(activation_memory, 0.3)
    
    # Total memory per iteration (forward + backward)
    # Backward pass typically uses 2-3x forward pass memory
    total_memory = model_memory + gradient_memory + (activation_memory * 3)
    
    return total_memory


def estimate_cost_per_epoch(
    memory_per_iteration: float,
    num_samples: int,
    batch_size: int = 1
) -> float:
    """
    Estimate cost per epoch in GPU-hours.
    Rough approximation: assumes time scales with memory usage.
    Based on empirical observations: ~0.12-0.15 seconds per iteration per GB of memory
    for 3D medical image CNNs with batch_size=1.
    """
    iterations_per_epoch = num_samples // batch_size
    # Time per iteration scales with memory, but with diminishing returns
    # For small memory (<1GB): ~0.15s per GB
    # For larger memory (>2GB): ~0.10s per GB
    if memory_per_iteration < 1.0:
        time_per_iteration = memory_per_iteration * 0.15  # seconds
    elif memory_per_iteration < 2.0:
        time_per_iteration = 0.15 + (memory_per_iteration - 1.0) * 0.12  # seconds
    else:
        time_per_iteration = 0.27 + (memory_per_iteration - 2.0) * 0.10  # seconds
    
    time_per_epoch = iterations_per_epoch * time_per_iteration  # seconds
    gpu_hours_per_epoch = time_per_epoch / 3600
    
    return gpu_hours_per_epoch


def read_mean_dimensions_from_statistics(statistics_file: str) -> Optional[Tuple[int, int, int]]:
    """
    Read mean height, width, and depth from statistics.txt file.
    
    Args:
        statistics_file: Path to statistics.txt file
        
    Returns:
        tuple: (height, width, depth) as integers, or None if not found
    """
    if not os.path.exists(statistics_file):
        return None
    
    try:
        with open(statistics_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse mean values: "Height range: 192.0 - 560.0 (mean: 430.3)" or "Height range: 192.0 - 560.0 (mean: 430.3, p95: 500.0, p99: 550.0)"
        # The pattern matches mean value even if followed by p95/p99
        height_match = re.search(r"Height range:[\d.\s-]+\(mean:\s*([\d.]+)", content)
        width_match = re.search(r"Width range:[\d.\s-]+\(mean:\s*([\d.]+)", content)
        depth_match = re.search(r"Depth range:[\d.\s-]+\(mean:\s*([\d.]+)", content)
        
        if height_match and width_match and depth_match:
            height = int(float(height_match.group(1)))
            width = int(float(width_match.group(1)))
            depth = int(float(depth_match.group(1)))
            return height, width, depth
    except Exception as e:
        print(f"    Warning: Could not read mean dimensions from {statistics_file}: {e}")
    
    return None


def read_num_samples_from_statistics(statistics_file: str) -> Optional[int]:
    """
    Read total number of samples from statistics.txt file (from Class Distribution section).
    
    Args:
        statistics_file: Path to statistics.txt file
        
    Returns:
        int: Total number of samples, or None if not found
    """
    if not os.path.exists(statistics_file):
        return None
    
    try:
        with open(statistics_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse class distribution: "Class 0: 92 samples (49.5%)"
        # Sum up all samples
        total_samples = 0
        pattern = r"Class\s+\d+:\s*(\d+)\s+samples"
        matches = re.findall(pattern, content)
        
        for match in matches:
            total_samples += int(match)
        
        if total_samples > 0:
            return total_samples
    except Exception as e:
        print(f"    Warning: Could not read num_samples from {statistics_file}: {e}")
    
    return None


def get_dataset_num_samples(dataset_name: str, voxel_calculation: str, data_path: str) -> int:
    """
    Get number of samples for a dataset from statistics.txt file.
    Falls back to estimates if file not found.
    """
    # Try preprocessed path first
    statistics_file = os.path.join(
        data_path,
        f"{dataset_name}_cleaned",
        f"preprocessed_{voxel_calculation}",
        "statistics.txt"
    )
    
    num_samples = read_num_samples_from_statistics(statistics_file)
    
    if num_samples is None:
        # Try alternative path without "preprocessed_" prefix
        statistics_file = os.path.join(
            data_path,
            f"{dataset_name}_cleaned",
            voxel_calculation,
            "statistics.txt"
        )
        num_samples = read_num_samples_from_statistics(statistics_file)
    
    if num_samples is None:
        # Fallback to estimates
        estimates = {
            "gist": 200,
            "lipo": 150,
            "melanoma": 100,
            "liver": 186,
            "desmoid": 203,
            "crlm": 150,
        }
        num_samples = estimates.get(dataset_name, 100)
        print(f"    Warning: Using estimate for num_samples: {num_samples}")
    
    return num_samples


def analyze_combination(
    dataset_name: str,
    model_type: str,
    voxel_calculation: str,
    data_path: str,
    total_trainings_per_config: int
) -> Dict:
    """
    Analyze a single combination and return cost estimate.
    """
    print(f"  Analyzing: {dataset_name} + {model_type} + {voxel_calculation}...")
    
    result = {
        "dataset": dataset_name,
        "model": model_type,
        "voxel_calculation": voxel_calculation,
        "spatial_size": None,
        "num_params": 0,
        "num_samples": 0,
        "memory_per_iteration_gb": 0.0,
        "cost_per_training_gpu_hours": 0.0,
        "cost_50_epochs_gpu_hours": 0.0,
        "error": None
    }
    if dataset_name in ["organmnist3d", "nodulemnist3d", "adrenalmnist3d", "fracturemnist3d", "vesselmnist3d", "synapsemnist3d"]:
        is_medmnist = True
    else:
        is_medmnist = False
    
    try:
        # Extract spatial_size
        if model_type in ["vit", "swin_unetr"]:
            # For ViT and SwinUNETR, use extract_spatial_size with maximum dimensions
            spatial_size = extract_spatial_size(
                model_type=model_type,
                voxel_calculation=voxel_calculation,
                dataset_name=dataset_name,
                developer_mode=False,
                data_path=data_path,
                is_medmnist=is_medmnist,
                use_percentile=False  # Use maximum dimensions
            )
            result["spatial_size"] = spatial_size
        else:
            # For other models (CNNs), use mean dimensions from statistics.txt
            statistics_file = os.path.join(
                data_path,
                f"{dataset_name}_cleaned",
                f"preprocessed_{voxel_calculation}",
                "statistics.txt"
            )
            
            spatial_size = read_mean_dimensions_from_statistics(statistics_file)
            
            if spatial_size is None:
                # Try alternative path without "preprocessed_" prefix
                statistics_file = os.path.join(
                    data_path,
                    f"{dataset_name}_cleaned",
                    voxel_calculation,
                    "statistics.txt"
                )
                spatial_size = read_mean_dimensions_from_statistics(statistics_file)
            
            result["spatial_size"] = spatial_size
        
        # Load search spaces
        if model_type not in SEARCH_SPACES:
            result["error"] = f"Model {model_type} not in SEARCH_SPACES"
            return result
        
        model_space = load_search_space(SEARCH_SPACES[model_type])
        training_space = load_search_space(TRAINING_SPACE)
        
        # Get max hyperparameters (pass model_type for special handling)
        model_hps = get_max_hyperparameters(model_space, model_type=model_type)
        training_hps = get_max_hyperparameters(training_space)
        all_hps = {**model_hps, **training_hps}
        
        # Check if spatial_size is needed and available
        if model_type in ["vit", "swin_unetr"] and spatial_size is None:
            # Skip if spatial_size couldn't be extracted for ViT/SwinUNETR
            result["error"] = "Could not extract spatial_size"
            return result
            
        num_params = estimate_model_parameters(
            model_type=model_type,
            hyperparameters=all_hps,
            spatial_size=spatial_size,
            num_classes=NUM_CLASSES,
            is_medmnist=is_medmnist
        )
        result["num_params"] = num_params
        
        if num_params == 0:
            result["error"] = "Could not estimate model parameters"
            return result
        
        # Estimate memory per iteration
        memory_per_iter = estimate_memory_per_iteration(
            model_type=model_type,
            num_params=num_params,
            spatial_size=spatial_size,
            batch_size=1
        )
        result["memory_per_iteration_gb"] = memory_per_iter
        
        # Estimate cost for 50 epochs
        num_samples = get_dataset_num_samples(dataset_name, voxel_calculation, data_path)
        cost_per_epoch = estimate_cost_per_epoch(
            memory_per_iteration=memory_per_iter,
            num_samples=num_samples,
            batch_size=1
        )
        # Cost per config = cost per epoch * epochs * total trainings (outer folds * inner folds)
        # Each outer fold trains cv_inner_folds models, so we multiply by total_trainings_per_config
        cost_per_training = cost_per_epoch * EPOCHS
        cost_per_config = cost_per_training * total_trainings_per_config
        result["cost_50_epochs_gpu_hours"] = cost_per_config
        result["cost_per_training_gpu_hours"] = cost_per_training
        result["num_samples"] = num_samples
        
    except Exception as e:
        result["error"] = str(e)
        print(f"    Error: {e}")
    
    return result


def main():
    """Main analysis function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze most expensive dataset-model-voxel_calculation combinations"
    )
    parser.add_argument(
        "--cv-outer-folds-repeats",
        type=int,
        default=None,
        help="Number of repetitions for outer CV folds (default: from experimental_setting.yaml)"
    )
    parser.add_argument(
        "--cv-outer-folds-splits",
        type=int,
        default=None,
        help="Number of splits per repetition for outer CV folds (default: from experimental_setting.yaml)"
    )
    parser.add_argument(
        "--cv-inner-folds",
        type=int,
        default=None,
        help="Number of inner CV folds (default: from experimental_setting.yaml)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top most expensive configurations to print (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ANALYZING MOST EXPENSIVE CONFIGURATIONS")
    print("=" * 80)
    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Voxel calculations: {VOXEL_CALCULATIONS}")
    print(f"Epochs: {EPOCHS}")
    
    # Load CV settings (use command line args if provided, otherwise from YAML)
    cv_settings = load_cv_settings(
        EXPERIMENTAL_SETTING,
        cv_outer_folds_repeats=args.cv_outer_folds_repeats,
        cv_outer_folds_splits=args.cv_outer_folds_splits,
        cv_inner_folds=args.cv_inner_folds
    )
    total_outer_folds = cv_settings["total_outer_folds"]
    cv_inner_folds = cv_settings["cv_inner_folds"]
    total_trainings = cv_settings["total_trainings_per_config"]
    
    print(f"CV settings: {cv_settings['cv_outer_folds_repeats']} repeats × {cv_settings['cv_outer_folds_splits']} splits = {total_outer_folds} outer folds")
    print(f"CV inner folds: {cv_inner_folds}")
    print(f"Total trainings per config: {total_outer_folds} outer × {cv_inner_folds} inner = {total_trainings}")
    print("=" * 80)
    print()
    
    results = []
    
    # Iterate through all combinations
    total_combinations = len(DATASETS) * len(MODELS) * len(VOXEL_CALCULATIONS)
    current = 0
    
    for dataset_name in DATASETS:
        for model_type in MODELS:
            for voxel_calculation in VOXEL_CALCULATIONS:
                current += 1
                print(f"[{current}/{total_combinations}] ", end="")
                
                result = analyze_combination(
                    dataset_name=dataset_name,
                    model_type=model_type,
                    voxel_calculation=voxel_calculation,
                    data_path=DATA_PATH,
                    total_trainings_per_config=total_trainings
                )
                results.append(result)
                print()
    
    # Filter out errors
    valid_results = [r for r in results if r["error"] is None]
    error_results = [r for r in results if r["error"] is not None]
    
    # Sort by cost
    valid_results.sort(key=lambda x: x["cost_50_epochs_gpu_hours"], reverse=True)
    
    # Print summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total combinations analyzed: {len(results)}")
    print(f"Valid combinations: {len(valid_results)}")
    print(f"Error combinations: {len(error_results)}")
    print()
    
    # Print top N most expensive
    top_n = args.top_n
    print(f"TOP {top_n} MOST EXPENSIVE CONFIGURATIONS (per config with CV):")
    print(f"Note: Cost includes {total_outer_folds} outer folds × {cv_inner_folds} inner folds = {total_trainings} trainings per config")
    print("-" * 120)
    print(f"{'Rank':<6} {'Dataset':<12} {'Model':<15} {'Voxel':<20} {'Spatial Size':<20} {'Samples':<8} {'Params (M)':<12} {'Cost/Config (GPU-h)':<20}")
    print("-" * 120)
    
    for i, result in enumerate(valid_results[:top_n], 1):
        spatial_str = str(result["spatial_size"]) if result["spatial_size"] else "N/A"
        params_m = result["num_params"] / 1e6
        cost = result["cost_50_epochs_gpu_hours"]
        num_samples = result.get("num_samples", 0)
        
        print(f"{i:<6} {result['dataset']:<12} {result['model']:<15} {result['voxel_calculation']:<20} "
              f"{spatial_str:<20} {num_samples:<8} {params_m:>10.2f}M {cost:>18.2f}")
    
    # Print errors if any
    if error_results:
        print()
        print("ERRORS:")
        print("-" * 80)
        for result in error_results:
            print(f"{result['dataset']} + {result['model']} + {result['voxel_calculation']}: {result['error']}")
    
    # Save results to file
    output_file = project_root / "analysis" / "most_expensive_configs.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MOST EXPENSIVE CONFIGURATIONS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Epochs per training: {EPOCHS}\n")
        f.write(f"CV settings: {cv_settings['cv_outer_folds_repeats']} repeats × {cv_settings['cv_outer_folds_splits']} splits = {total_outer_folds} outer folds\n")
        f.write(f"CV inner folds: {cv_inner_folds}\n")
        f.write(f"Total trainings per config: {total_trainings}\n")
        f.write(f"Total combinations: {len(results)}\n")
        f.write(f"Valid: {len(valid_results)}\n")
        f.write(f"Errors: {len(error_results)}\n")
        f.write(f"Top N printed: {args.top_n}\n\n")
        
        f.write("ALL RESULTS (sorted by cost, descending):\n")
        f.write("-" * 140 + "\n")
        f.write(f"{'Dataset':<12} {'Model':<15} {'Voxel':<20} {'Spatial Size':<20} {'Samples':<8} "
                f"{'Params (M)':<12} {'Memory/Iter (GB)':<18} {'Cost/Training (GPU-h)':<20} {'Cost/Config (GPU-h)':<20}\n")
        f.write("-" * 140 + "\n")
        
        for result in valid_results:
            spatial_str = str(result["spatial_size"]) if result["spatial_size"] else "N/A"
            params_m = result["num_params"] / 1e6
            memory = result["memory_per_iteration_gb"]
            cost_per_training = result.get("cost_per_training_gpu_hours", 0.0)
            cost_per_config = result["cost_50_epochs_gpu_hours"]
            num_samples = result.get("num_samples", 0)
            
            f.write(f"{result['dataset']:<12} {result['model']:<15} {result['voxel_calculation']:<20} "
                    f"{spatial_str:<20} {num_samples:<8} {params_m:>10.2f}M {memory:>16.2f} "
                    f"{cost_per_training:>18.2f} {cost_per_config:>18.2f}\n")
        
        if error_results:
            f.write("\nERRORS:\n")
            f.write("-" * 80 + "\n")
            for result in error_results:
                f.write(f"{result['dataset']} + {result['model']} + {result['voxel_calculation']}: {result['error']}\n")
    
    print()
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

