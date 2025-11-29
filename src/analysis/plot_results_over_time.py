"""
Plot test and validation performance over time (number of configs) for NePS experiments.

This script reads NePS experiment results and creates plots showing:
- Test performance (AUC) over time
- Validation performance (AUC) over time

Both plots show mean ± std across all outer cross-validation folds.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def extract_config_number(config_name: str) -> int:
    """Extract config number from config name (e.g., 'config_3' -> 3)."""
    return int(config_name.split("_")[-1])


def get_experiment_name_with_prefix(experiment_dir: Path) -> str:
    """
    Get experiment name with appropriate prefix (Baseline_ or NePS_) based on path.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Experiment name with prefix (e.g., "Baseline_test_liver_33" or "NePS_test_plotting_script_7")
    """
    experiment_name = experiment_dir.name
    experiment_path_str = str(experiment_dir)
    
    # Check if it's a Baseline experiment (path contains "experiments/Baseline/")
    if "/Baseline/" in experiment_path_str or "\\Baseline\\" in experiment_path_str:
        return f"Baseline_{experiment_name}"
    # Check if it's a NePS experiment (path contains "experiments/NePS/")
    elif "/NePS/" in experiment_path_str or "\\NePS\\" in experiment_path_str:
        return f"NePS_{experiment_name}"
    else:
        # Fallback: return name without prefix if path doesn't match expected pattern
        return experiment_name


def load_validation_performance(report_path: Path) -> float:
    """
    Load validation performance from report.yaml.
    
    Args:
        report_path: Path to report.yaml file
        
    Returns:
        Validation performance (objective_to_minimize without negative sign)
    """
    try:
        with open(report_path, "r") as f:
            report = yaml.safe_load(f)
        
        objective = report.get("objective_to_minimize", None)
        if objective is None:
            raise ValueError(f"No 'objective_to_minimize' found in {report_path}")
        
        # Remove negative sign as requested
        return abs(objective)
    except Exception as e:
        raise ValueError(f"Error loading validation performance from {report_path}: {e}")


def load_test_performance(results_path: Path) -> float:
    """
    Load test performance from test_evaluation_results.json.
    
    Args:
        results_path: Path to test_evaluation_results.json file
        
    Returns:
        Test performance (auc_macro from ensemble)
    """
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        
        ensemble = results.get("ensemble", None)
        if ensemble is None:
            raise ValueError(f"No 'ensemble' key found in {results_path}")
        
        auc_macro = ensemble.get("auc_macro", None)
        if auc_macro is None:
            raise ValueError(f"No 'auc_macro' found in ensemble in {results_path}")
        
        return float(auc_macro)
    except Exception as e:
        raise ValueError(f"Error loading test performance from {results_path}: {e}")


def collect_performances(experiment_dir: Path) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Collect validation and test performances across all outer folds.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., experiments/NePS/lipo/test_plotting_script)
        
    Returns:
        Tuple of (validation_performances, test_performances)
        Each is a dict mapping config_number -> list of performances across outer folds
        Performances are stored per fold, not aggregated yet
    """
    # Store performances per fold: fold_index -> list of (config_num, performance)
    validation_performances_per_fold: Dict[int, List[Tuple[int, float]]] = {}
    test_performances_per_fold: Dict[int, List[Tuple[int, float]]] = {}
    
    # Find all seed directories
    seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    
    if not seed_dirs:
        raise ValueError(f"No seed directories found in {experiment_dir}")
    
    print(f"Found {len(seed_dirs)} seed directory/ies")
    
    fold_counter = 0
    
    # Process each seed
    for seed_dir in seed_dirs:
        neps_output_dir = seed_dir / "NePS_output"
        
        if not neps_output_dir.exists():
            print(f"Warning: NePS_output directory not found in {seed_dir}, skipping...")
            continue
        
        # Find all outer fold directories
        outer_fold_dirs = sorted(
            [d for d in neps_output_dir.iterdir() 
             if d.is_dir() and d.name.startswith("cv_outer_fold_")]
        )
        
        if not outer_fold_dirs:
            print(f"Warning: No outer fold directories found in {neps_output_dir}, skipping...")
            continue
        
        print(f"  Processing {len(outer_fold_dirs)} outer fold(s) in {seed_dir.name}")
        
        # Process each outer fold
        for outer_fold_dir in outer_fold_dirs:
            configs_dir = outer_fold_dir / "configs"
            
            if not configs_dir.exists():
                print(f"Warning: configs directory not found in {outer_fold_dir}, skipping...")
                continue
            
            # Find all config directories
            config_dirs = sorted(
                [d for d in configs_dir.iterdir() 
                 if d.is_dir() and d.name.startswith("config_")],
                key=lambda x: extract_config_number(x.name)
            )
            
            # Initialize lists for this fold
            validation_performances_per_fold[fold_counter] = []
            test_performances_per_fold[fold_counter] = []
            
            # Process each config
            for config_dir in config_dirs:
                config_num = extract_config_number(config_dir.name)
                
                # Load validation performance
                # Try report.yaml first (created by NePS after run_pipeline returns)
                # If not available, try pipeline_result.json (created during run_pipeline)
                val_perf = None
                report_path = config_dir / "report.yaml"
                if report_path.exists():
                    try:
                        val_perf = load_validation_performance(report_path)
                        # Debug: Print loaded validation performance
                        print(f"    Config {config_num}: Validation performance = {val_perf:.2f} (from report.yaml)")
                    except Exception as e:
                        print(f"Warning: Could not load validation performance from {report_path}: {e}")
                
                # Fallback: Try pipeline_result.json if report.yaml doesn't exist or failed
                if val_perf is None:
                    pipeline_result_path = config_dir / "pipeline_result.json"
                    if pipeline_result_path.exists():
                        try:
                            with open(pipeline_result_path, "r", encoding="utf-8") as f:
                                pipeline_result = json.load(f)
                            objective = pipeline_result.get("objective_to_minimize", None)
                            if objective is not None:
                                val_perf = abs(objective)
                                print(f"    Config {config_num}: Validation performance = {val_perf:.2f} (from pipeline_result.json)")
                        except Exception as e:
                            print(f"Warning: Could not load validation performance from {pipeline_result_path}: {e}")
                
                if val_perf is not None:
                    validation_performances_per_fold[fold_counter].append((config_num, val_perf))
                
                # Load test performance
                test_results_path = config_dir / "test_evaluation_results.json"
                if test_results_path.exists():
                    try:
                        test_perf = load_test_performance(test_results_path)
                        # Debug: Print loaded test performance
                        print(f"    Config {config_num}: Test performance = {test_perf:.2f}")
                        test_performances_per_fold[fold_counter].append((config_num, test_perf))
                    except Exception as e:
                        print(f"Warning: Could not load test performance from {test_results_path}: {e}")
            
            fold_counter += 1
    
    # Calculate incumbent performances for each fold independently
    validation_performances: Dict[int, List[float]] = {}
    test_performances: Dict[int, List[float]] = {}
    
    # Process each fold independently
    for fold_idx in sorted(validation_performances_per_fold.keys()):
        val_perfs = validation_performances_per_fold.get(fold_idx, [])
        test_perfs = test_performances_per_fold.get(fold_idx, [])
        
        if not val_perfs:
            print(f"Warning: No validation performances for fold {fold_idx}, skipping...")
            continue
        
        # Sort by config number for this fold
        val_perfs_sorted = sorted(val_perfs, key=lambda x: x[0])
        test_perfs_dict = dict(test_perfs) if test_perfs else {}
        
        print(f"Processing fold {fold_idx}: {len(val_perfs_sorted)} configs")
        
        # Calculate validation incumbent for this fold (best validation so far)
        best_val_so_far = float('-inf')
        for config_num, val_perf in val_perfs_sorted:
            best_val_so_far = max(best_val_so_far, val_perf)  # For AUC, higher is better
            if config_num not in validation_performances:
                validation_performances[config_num] = []
            validation_performances[config_num].append(best_val_so_far)
            print(f"  Fold {fold_idx}, Config {config_num}: Val={val_perf:.2f}, Incumbent={best_val_so_far:.2f}")
        
        # For test performances: use test performance of config with best validation so far (for this fold)
        best_val_so_far = float('-inf')
        best_val_config = None
        
        for config_num, val_perf in val_perfs_sorted:
            # Update best validation performance (incumbent) for this fold
            if val_perf > best_val_so_far:
                best_val_so_far = val_perf
                best_val_config = config_num
            
            # Use test performance of the config with best validation performance so far (in this fold)
            if best_val_config is not None and best_val_config in test_perfs_dict:
                test_perf_to_use = test_perfs_dict[best_val_config]
                if config_num not in test_performances:
                    test_performances[config_num] = []
                test_performances[config_num].append(test_perf_to_use)
    
    # Fill missing configs with last incumbent value for each fold
    # Find maximum config number across all folds
    max_config = 0
    if validation_performances:
        max_config = max(max_config, max(validation_performances.keys()))
    if test_performances:
        max_config = max(max_config, max(test_performances.keys()))
    
    # For each fold, fill missing configs with last incumbent value
    num_folds = len(validation_performances_per_fold)
    for fold_idx in range(num_folds):
        # Find the last config that exists in this fold and its values
        # We need to find the last config that has data for this specific fold
        last_val_incumbent = None
        last_test_value = None
        
        # Find last config with data in this fold (iterate in reverse to get the last one)
        for config_num in sorted(validation_performances.keys(), reverse=True):
            if config_num in validation_performances and fold_idx < len(validation_performances[config_num]):
                val_value = validation_performances[config_num][fold_idx]
                if val_value is not None:
                    last_val_incumbent = val_value
                    break
        
        for config_num in sorted(test_performances.keys(), reverse=True):
            if config_num in test_performances and fold_idx < len(test_performances[config_num]):
                test_value = test_performances[config_num][fold_idx]
                if test_value is not None:
                    last_test_value = test_value
                    break
        
        # Fill missing configs with last incumbent values
        for config_num in range(1, max_config + 1):
            # Fill validation performances
            if config_num not in validation_performances:
                validation_performances[config_num] = [None] * num_folds
            # Ensure list is long enough
            while len(validation_performances[config_num]) < num_folds:
                validation_performances[config_num].append(None)
            
            # If this config is missing in this fold, use last incumbent value
            if validation_performances[config_num][fold_idx] is None:
                if last_val_incumbent is not None:
                    validation_performances[config_num][fold_idx] = last_val_incumbent
            
            # Fill test performances
            if config_num not in test_performances:
                test_performances[config_num] = [None] * num_folds
            # Ensure list is long enough
            while len(test_performances[config_num]) < num_folds:
                test_performances[config_num].append(None)
            
            # If this config is missing in this fold, use last test value
            if test_performances[config_num][fold_idx] is None:
                if last_test_value is not None:
                    test_performances[config_num][fold_idx] = last_test_value
    
    # Debug: Print collected performances
    print("\nCollected validation performances:")
    for config_num in sorted(validation_performances.keys()):
        print(f"  Config {config_num}: {validation_performances[config_num]}")
    
    # Save performances to CSV file
    save_performances_to_csv(experiment_dir, validation_performances, test_performances)
    
    return validation_performances, test_performances


def save_performances_to_csv(
    experiment_dir: Path,
    validation_performances: Dict[int, List[float]],
    test_performances: Dict[int, List[float]]
) -> None:
    """
    Save validation and test performances to a CSV file.
    
    The CSV file contains:
    - config: Config number
    - validation_fold_0, validation_fold_1, ...: Validation incumbent performances per fold
    - test_fold_0, test_fold_1, ...: Test performances (of config with best validation) per fold
    - validation_mean, validation_std: Mean and std across folds
    - test_mean, test_std: Mean and std across folds
    
    Args:
        experiment_dir: Path to experiment directory
        validation_performances: Dict mapping config_number -> list of validation incumbent performances across folds
        test_performances: Dict mapping config_number -> list of test performances across folds
    """
    # Find seed directory to determine output path
    seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        print("Warning: No seed directories found, cannot save incumbent_performances.csv")
        return
    
    # Use first seed directory (or iterate through all if needed)
    seed_dir = seed_dirs[0]
    neps_output_dir = seed_dir / "NePS_output"
    
    if not neps_output_dir.exists():
        print(f"Warning: NePS_output directory not found: {neps_output_dir}")
        return
    
    # Path to incumbent_performances.csv in NePS output directory
    performances_csv_path = neps_output_dir / "incumbent_performances.csv"
    
    # Collect all data for CSV
    csv_rows = []
    
    # Get all config numbers
    all_configs = sorted(set(list(validation_performances.keys()) + list(test_performances.keys())))
    
    # Get maximum number of folds (for proper alignment)
    max_folds = 0
    for config_num in all_configs:
        val_folds = len(validation_performances.get(config_num, []))
        test_folds = len(test_performances.get(config_num, []))
        max_folds = max(max_folds, val_folds, test_folds)
    
    # Create CSV header
    header = ["config"]
    for fold_idx in range(max_folds):
        header.append(f"validation_fold_{fold_idx}")
    for fold_idx in range(max_folds):
        header.append(f"test_fold_{fold_idx}")
    header.extend(["validation_mean", "validation_std", "test_mean", "test_std"])
    
    csv_rows.append(header)
    
    # Data rows
    for config_num in all_configs:
        row = [config_num]
        
        # Validation performances for this config
        val_perfs = validation_performances.get(config_num, [])
        for fold_idx in range(max_folds):
            if fold_idx < len(val_perfs):
                row.append(val_perfs[fold_idx])
            else:
                row.append("")  # Empty if no data for this fold
        
        # Test performances for this config
        test_perfs = test_performances.get(config_num, [])
        for fold_idx in range(max_folds):
            if fold_idx < len(test_perfs):
                row.append(test_perfs[fold_idx])
            else:
                row.append("")  # Empty if no data for this fold
        
        # Calculate mean and std
        val_mean = np.mean(val_perfs) if val_perfs else ""
        val_std = np.std(val_perfs) if val_perfs and len(val_perfs) > 1 else (0.0 if val_perfs else "")
        test_mean = np.mean(test_perfs) if test_perfs else ""
        test_std = np.std(test_perfs) if test_perfs and len(test_perfs) > 1 else (0.0 if test_perfs else "")
        
        row.extend([val_mean, val_std, test_mean, test_std])
        
        csv_rows.append(row)
    
    # Write CSV file
    import csv as csv_module
    with open(performances_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        writer.writerows(csv_rows)
    
    print(f"\nSaved incumbent performances to: {performances_csv_path}")


def calculate_mean_std(performances: Dict[int, List[float]]) -> Tuple[List[int], List[float], List[float]]:
    """
    Calculate mean and std for each config number.
    
    Args:
        performances: Dict mapping config_number -> list of performances
        
    Returns:
        Tuple of (config_numbers, means, stds) as sorted lists
    """
    config_numbers = sorted(performances.keys())
    means = []
    stds = []
    
    for config_num in config_numbers:
        perfs = performances[config_num]
        means.append(np.mean(perfs))
        stds.append(np.std(perfs))
    
    return config_numbers, means, stds


def extend_performances_to_max_configs(
    performances: Dict[int, List[float]],
    max_config: int
) -> Dict[int, List[float]]:
    """
    Extend performances dictionary to max_config by repeating the last performance value.
    
    Args:
        performances: Dictionary mapping config_num to list of performance values
        max_config: Maximum config number to extend to
        
    Returns:
        Extended performances dictionary
    """
    if not performances:
        return performances
    
    # Find the maximum config number in the current performances
    current_max_config = max(performances.keys())
    
    # If already at or above max_config, return as is
    if current_max_config >= max_config:
        return performances
    
    # Get the last performance value (for each fold)
    last_performance = performances[current_max_config]
    
    # Extend by repeating the last performance for all missing configs
    extended = performances.copy()
    for config_num in range(current_max_config + 1, max_config + 1):
        extended[config_num] = last_performance
    
    return extended


def create_plots(
    experiment_dirs: List[Path],
    all_validation_performances: List[Tuple[str, Dict[int, List[float]]]],
    all_test_performances: List[Tuple[str, Dict[int, List[float]]]],
    output_path: Path = None,
    extend_to_max_configs: bool = False
):
    """
    Create plots for validation and test performance over time.
    
    Args:
        experiment_dirs: List of paths to experiment directories
        all_validation_performances: List of tuples (experiment_name, validation_performances dict)
        all_test_performances: List of tuples (experiment_name, test_performances dict)
        output_path: Optional path to save the plot. If None, saves to first seed directory.
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    
    # Define colors for different experiments
    experiment_colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_dirs)))
    validation_color_base = "blue"
    test_color_base = "green"
    
    # Collect all config numbers across all experiments for x-axis
    all_val_configs = set()
    all_test_configs = set()
    for _, val_perfs in all_validation_performances:
        all_val_configs.update(val_perfs.keys())
    for _, test_perfs in all_test_performances:
        all_test_configs.update(test_perfs.keys())
    
    all_configs = sorted(set(all_val_configs | all_test_configs))
    
    # If extend_to_max_configs is True, extend shorter experiments to match the longest one
    if extend_to_max_configs and len(experiment_dirs) > 1 and all_configs:
        max_config = max(all_configs)
        print(f"\nExtending experiments to max_config={max_config}...")
        
        # Extend validation and test performances
        extended_validation_performances = []
        extended_test_performances = []
        
        for exp_name, validation_perfs in all_validation_performances:
            extended_val = extend_performances_to_max_configs(validation_perfs, max_config)
            extended_validation_performances.append((exp_name, extended_val))
        
        for exp_name, test_perfs in all_test_performances:
            extended_test = extend_performances_to_max_configs(test_perfs, max_config)
            extended_test_performances.append((exp_name, extended_test))
        
        # Update the lists
        all_validation_performances = extended_validation_performances
        all_test_performances = extended_test_performances
        
        # Recalculate all_configs to include the extended range
        all_val_configs = set()
        all_test_configs = set()
        for _, val_perfs in all_validation_performances:
            all_val_configs.update(val_perfs.keys())
        for _, test_perfs in all_test_performances:
            all_test_configs.update(test_perfs.keys())
        all_configs = sorted(set(all_val_configs | all_test_configs))
    
    # Calculate global y-axis range across all plots for better comparison
    all_y_values = []
    for exp_name, validation_performances in all_validation_performances:
        if validation_performances:
            val_configs, val_means, val_stds = calculate_mean_std(validation_performances)
            if val_configs:
                # Debug: Print loaded validation values
                print(f"  Validation values for {exp_name}: means={val_means}, stds={val_stds}")
                # Only include reasonable values (AUC should be between 0 and 100)
                val_means_array = np.array(val_means)
                val_stds_array = np.array(val_stds)
                val_upper = val_means_array + val_stds_array
                val_lower = val_means_array - val_stds_array
                # Filter out clearly invalid values
                valid_mask = (val_lower >= 0) & (val_lower <= 100) & (val_upper >= 0) & (val_upper <= 100)
                if np.any(valid_mask):
                    all_y_values.extend(val_upper[valid_mask].tolist())
                    all_y_values.extend(val_lower[valid_mask].tolist())
                else:
                    # If all values are filtered out, use the means anyway (might be edge case)
                    all_y_values.extend(val_means)
    
    for exp_name, test_performances in all_test_performances:
        if test_performances:
            test_configs, test_means, test_stds = calculate_mean_std(test_performances)
            if test_configs:
                # Debug: Print loaded test values
                print(f"  Test values for {exp_name}: means={test_means}, stds={test_stds}")
                # Only include reasonable values (AUC should be between 0 and 100)
                test_means_array = np.array(test_means)
                test_stds_array = np.array(test_stds)
                test_upper = test_means_array + test_stds_array
                test_lower = test_means_array - test_stds_array
                # Filter out clearly invalid values
                valid_mask = (test_lower >= 0) & (test_lower <= 100) & (test_upper >= 0) & (test_upper <= 100)
                if np.any(valid_mask):
                    all_y_values.extend(test_upper[valid_mask].tolist())
                    all_y_values.extend(test_lower[valid_mask].tolist())
                else:
                    # If all values are filtered out, use the means anyway (might be edge case)
                    all_y_values.extend(test_means)
    
    # Set y-axis range with some padding
    # Y-axis should start at 50, unless performance goes below 50, then adjust downward
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.1  # 10% padding
        
        # Start at 50 if all values are >= 50, otherwise start below 50
        if y_min >= 50:
            y_lim_min = 50
        else:
            y_lim_min = max(0, y_min - y_padding)  # Don't go below 0 for AUC
        
        y_lim_max = 100  # Always fix upper limit at 100% for AUC
        
        print(f"Y-axis range: min={y_min:.2f}, max={y_max:.2f}, y_lim_min={y_lim_min:.2f}, y_lim_max={y_lim_max:.2f}")
    else:
        y_lim_min = 50
        y_lim_max = 100
        print(f"No y-values found, using default range: y_lim_min={y_lim_min}, y_lim_max={y_lim_max}")
    
    # Plot validation performance for each experiment
    has_validation_data = False
    for idx, (exp_name, validation_performances) in enumerate(all_validation_performances):
        if validation_performances:
            val_configs, val_means, val_stds = calculate_mean_std(validation_performances)
            if val_configs:
                has_validation_data = True
                # Use different shades of blue for different experiments
                color = experiment_colors[idx] if len(experiment_dirs) > 1 else validation_color_base
                ax1.plot(val_configs, val_means, marker="o", linewidth=2, markersize=8, 
                        label=f"{exp_name} (Mean with std)", color=color, linestyle="--")
                ax1.fill_between(
                    val_configs,
                    np.array(val_means) - np.array(val_stds),
                    np.array(val_means) + np.array(val_stds),
                    alpha=0.2,
                    color=color
                )
    
    if has_validation_data:
        ax1.set_xlabel("Number of Configs", fontsize=12)
        ax1.set_ylabel("Validation AUC", fontsize=12)
        ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        if len(all_configs) == 1:
            ax1.set_xlim(left=0, right=all_configs[0] + 1)
            ax1.set_xticks([all_configs[0]])
        else:
            ax1.set_xticks(all_configs)
        # Set y-axis limits after all other operations
        ax1.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax1.text(0.5, 0.5, "No validation data available", 
                ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
        ax1.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Plot test performance for each experiment
    has_test_data = False
    for idx, (exp_name, test_performances) in enumerate(all_test_performances):
        if test_performances:
            test_configs, test_means, test_stds = calculate_mean_std(test_performances)
            if test_configs:
                has_test_data = True
                # Use different shades of green for different experiments
                color = experiment_colors[idx] if len(experiment_dirs) > 1 else test_color_base
                ax2.plot(test_configs, test_means, marker="s", linewidth=2, markersize=8, 
                        label=f"{exp_name} (Mean with std)", color=color, linestyle="-")
                ax2.fill_between(
                    test_configs,
                    np.array(test_means) - np.array(test_stds),
                    np.array(test_means) + np.array(test_stds),
                    alpha=0.2,
                    color=color
                )
    
    if has_test_data:
        ax2.set_xlabel("Number of Configs", fontsize=12)
        ax2.set_ylabel("Test AUC", fontsize=12)
        ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        if len(all_configs) == 1:
            ax2.set_xlim(left=0, right=all_configs[0] + 1)
            ax2.set_xticks([all_configs[0]])
        else:
            ax2.set_xticks(all_configs)
        # Set y-axis limits after all other operations
        ax2.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax2.text(0.5, 0.5, "No test data available", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
        ax2.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Plot both validation and test performance together for each experiment
    if has_validation_data or has_test_data:
        # Plot validation for each experiment
        for idx, (exp_name, validation_performances) in enumerate(all_validation_performances):
            if validation_performances:
                val_configs, val_means, val_stds = calculate_mean_std(validation_performances)
                if val_configs:
                    color = experiment_colors[idx] if len(experiment_dirs) > 1 else validation_color_base
                    ax3.plot(val_configs, val_means, marker="o", linewidth=2, markersize=8, 
                            label=f"{exp_name} (Validation)", color=color, linestyle="--")
                    ax3.fill_between(
                        val_configs,
                        np.array(val_means) - np.array(val_stds),
                        np.array(val_means) + np.array(val_stds),
                        alpha=0.15,
                        color=color
                    )
        
        # Plot test for each experiment
        for idx, (exp_name, test_performances) in enumerate(all_test_performances):
            if test_performances:
                test_configs, test_means, test_stds = calculate_mean_std(test_performances)
                if test_configs:
                    color = experiment_colors[idx] if len(experiment_dirs) > 1 else test_color_base
                    ax3.plot(test_configs, test_means, marker="s", linewidth=2, markersize=8, 
                            label=f"{exp_name} (Test)", color=color, linestyle="-")
                    ax3.fill_between(
                        test_configs,
                        np.array(test_means) - np.array(test_stds),
                        np.array(test_means) + np.array(test_stds),
                        alpha=0.15,
                        color=color
                    )
        
        ax3.set_xlabel("Number of Configs", fontsize=12)
        ax3.set_ylabel("AUC", fontsize=12)
        ax3.set_title("Validation & Test Performance Over Time", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=9, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=0)
        if len(all_configs) == 1:
            ax3.set_xlim(left=0, right=all_configs[0] + 1)
            ax3.set_xticks([all_configs[0]])
        else:
            ax3.set_xticks(all_configs)
        # Set y-axis limits after all other operations
        ax3.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax3.text(0.5, 0.5, "No data available", 
                ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Validation & Test Performance Over Time", fontsize=14, fontweight="bold")
        ax3.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Add experiment names as suptitle
    if len(experiment_dirs) == 1:
        experiment_name = get_experiment_name_with_prefix(experiment_dirs[0])
        fig.suptitle(f"Performance Over Time: {experiment_name}", fontsize=16, fontweight="bold", y=1.02)
    else:
        experiment_names = ", ".join([get_experiment_name_with_prefix(exp_dir) for exp_dir in experiment_dirs])
        fig.suptitle(f"Performance Over Time: {experiment_names}", fontsize=16, fontweight="bold", y=1.02)
    
    # Determine output path
    if output_path is None:
        # Use first experiment directory to save plots
        first_experiment_dir = experiment_dirs[0]
        seed_dirs = sorted([d for d in first_experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        if seed_dirs:
            output_path = seed_dirs[0] / "performance_over_time.png"
        else:
            # Fallback to experiment directory if no seed directory found
            output_path = first_experiment_dir / "performance_over_time.png"
    else:
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply tight layout and then re-apply y-axis limits (tight_layout might reset them)
    plt.tight_layout()
    
    # Re-apply y-axis limits after tight_layout (which might reset them)
    # Use autoscale=False to prevent matplotlib from auto-adjusting
    for ax in [ax1, ax2, ax3]:
        ax.set_autoscale_on(False)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    # Re-apply y-axis limits before saving PDF as well
    for ax in [ax1, ax2, ax3]:
        ax.set_autoscale_on(False)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plot saved to: {pdf_path}")
    
    plt.close()


def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot test and validation performance over time for NePS experiments. "
                    "Can plot single or multiple experiments together."
    )
    parser.add_argument(
        "experiment_dirs",
        type=str,
        nargs="+",
        help="Path(s) to experiment directory/ies (e.g., experiments/NePS/lipo/test_plotting_script). "
             "Can specify multiple experiments to compare them in one plot."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the plot. If not specified, saves to first experiment's seed directory."
    )
    parser.add_argument(
        "--extend-to-max-configs",
        action="store_true",
        help="If set, extend experiments with fewer configs to match the longest experiment by repeating the last performance value. "
             "Useful when comparing Baseline (1 config) with NePS runs (multiple configs)."
    )
    
    args = parser.parse_args()
    
    experiment_dirs = [Path(exp_dir) for exp_dir in args.experiment_dirs]
    
    # Validate all experiment directories
    for experiment_dir in experiment_dirs:
        if not experiment_dir.exists():
            raise ValueError(f"Experiment directory does not exist: {experiment_dir}")
    
    print(f"Processing {len(experiment_dirs)} experiment(s):")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir}")
    print("=" * 60)
    
    # Collect performances for each experiment
    all_validation_performances = []
    all_test_performances = []
    
    for experiment_dir in experiment_dirs:
        print(f"\nProcessing experiment: {experiment_dir.name}")
        validation_performances, test_performances = collect_performances(experiment_dir)
        # Get experiment name with appropriate prefix
        exp_name_with_prefix = get_experiment_name_with_prefix(experiment_dir)
        all_validation_performances.append((exp_name_with_prefix, validation_performances))
        all_test_performances.append((exp_name_with_prefix, test_performances))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    for idx, (exp_name, validation_perfs, test_perfs) in enumerate(
        zip([get_experiment_name_with_prefix(exp) for exp in experiment_dirs], 
            [vp[1] for vp in all_validation_performances],
            [tp[1] for tp in all_test_performances])
    ):
        print(f"\n  {exp_name}:")
        print(f"    Validation performances: {len(validation_perfs)} configs")
        print(f"    Test performances: {len(test_perfs)} configs")
    
    # Create plots
    output_path = Path(args.output) if args.output else None
    create_plots(
        experiment_dirs, 
        all_validation_performances, 
        all_test_performances, 
        output_path,
        extend_to_max_configs=args.extend_to_max_configs
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

