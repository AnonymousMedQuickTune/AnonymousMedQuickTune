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
    Get experiment name with appropriate prefix (Baseline_, NePS_, or QuickTune_) based on path.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Experiment name with prefix (e.g., "Baseline_test_liver_33", "NePS_test_plotting_script_7", or "QuickTune_test_experiment")
    """
    experiment_name = experiment_dir.name
    experiment_path_str = str(experiment_dir)
    
    # Check if it's a Baseline experiment (path contains "experiments/Baseline/")
    if "/Baseline/" in experiment_path_str or "\\Baseline\\" in experiment_path_str:
        return f"Baseline_{experiment_name}"
    # Check if it's a QuickTune experiment (path contains "experiments/QuickTune/" or "experiments/Cluster/QuickTune/")
    elif "/QuickTune/" in experiment_path_str or "\\QuickTune\\" in experiment_path_str:
        return f"QuickTune_{experiment_name}"
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


def load_validation_performance_quicktune(results_path: Path, metric: str = "auc") -> float:
    """
    Load validation performance from QuickTune test_evaluation_results.json.
    
    Args:
        results_path: Path to test_evaluation_results.json file
        metric: Metric to extract (default: "auc")
        
    Returns:
        Validation performance (from validation.{metric})
    """
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        
        validation = results.get("validation", None)
        if validation is None:
            raise ValueError(f"No 'validation' key found in {results_path}")
        
        # Try the specified metric, fallback to common metrics
        perf = validation.get(metric, None)
        if perf is None:
            # Try common alternatives
            if metric == "auc":
                perf = validation.get("auc", None)
            elif metric == "accuracy":
                perf = validation.get("accuracy", None)
            elif metric == "f1":
                perf = validation.get("f1", None)
        
        if perf is None:
            raise ValueError(f"No '{metric}' found in validation in {results_path}. Available keys: {list(validation.keys())}")
        
        return float(perf)
    except Exception as e:
        raise ValueError(f"Error loading validation performance from {results_path}: {e}")


def collect_performances_quicktune(experiment_dir: Path, metric: str = "auc") -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Collect validation and test performances across all outer folds for QuickTune experiments.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., experiments/QuickTune/lipo/test_experiment)
        metric: Metric to use for validation performance (default: "auc")
        
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
        # Find all outer fold directories
        outer_fold_dirs = sorted(
            [d for d in seed_dir.iterdir() 
             if d.is_dir() and d.name.startswith("cv_outer_fold_")]
        )
        
        if not outer_fold_dirs:
            print(f"Warning: No outer fold directories found in {seed_dir}, skipping...")
            continue
        
        print(f"  Processing {len(outer_fold_dirs)} outer fold(s) in {seed_dir.name}")
        
        # Process each outer fold
        for outer_fold_dir in outer_fold_dirs:
            tuner_dir = outer_fold_dir / "tuner"
            
            if not tuner_dir.exists():
                print(f"Warning: tuner directory not found in {outer_fold_dir}, skipping...")
                continue
            
            # Find all config directories (numeric directories in tuner/)
            config_dirs = []
            for item in tuner_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    try:
                        config_num = int(item.name)
                        config_dirs.append((config_num, item))
                    except ValueError:
                        continue
            
            # Sort by config number
            config_dirs = sorted(config_dirs, key=lambda x: x[0])
            
            if not config_dirs:
                print(f"Warning: No config directories found in {tuner_dir}, skipping...")
                continue
            
            # Initialize lists for this fold
            validation_performances_per_fold[fold_counter] = []
            test_performances_per_fold[fold_counter] = []
            
            # Process each config
            for config_num, config_dir in config_dirs:
                # Load validation performance from test_evaluation_results.json
                test_results_path = config_dir / "test_evaluation_results.json"
                if not test_results_path.exists():
                    print(f"Warning: test_evaluation_results.json not found in {config_dir}, skipping config {config_num}...")
                    continue
                
                # Load validation performance
                val_perf = None
                try:
                    val_perf = load_validation_performance_quicktune(test_results_path, metric)
                    print(f"    Config {config_num}: Validation performance = {val_perf:.2f} (from test_evaluation_results.json, metric={metric})")
                except Exception as e:
                    print(f"Warning: Could not load validation performance from {test_results_path}: {e}")
                    continue
                
                if val_perf is not None:
                    validation_performances_per_fold[fold_counter].append((config_num, val_perf))
                
                # Load test performance
                try:
                    test_perf = load_test_performance(test_results_path)
                    print(f"    Config {config_num}: Test performance = {test_perf:.2f}")
                    test_performances_per_fold[fold_counter].append((config_num, test_perf))
                except Exception as e:
                    print(f"Warning: Could not load test performance from {test_results_path}: {e}")
            
            fold_counter += 1
    
    # Calculate incumbent performances for each fold independently (same logic as NePS)
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
            
            # Use test performance of the best validation config so far
            if best_val_config is not None and best_val_config in test_perfs_dict:
                test_perf = test_perfs_dict[best_val_config]
                if config_num not in test_performances:
                    test_performances[config_num] = []
                test_performances[config_num].append(test_perf)
    
    # Fill missing configs with last incumbent values (same logic as NePS)
    num_folds = len(validation_performances_per_fold)
    max_config = max(validation_performances.keys()) if validation_performances else 0
    
    for fold_idx in range(num_folds):
        last_val_incumbent = None
        last_test_value = None
        
        # Find last incumbent values for this fold
        for config_num in sorted(validation_performances.keys()):
            if fold_idx < len(validation_performances[config_num]):
                if validation_performances[config_num][fold_idx] is not None:
                    last_val_incumbent = validation_performances[config_num][fold_idx]
            if config_num in test_performances and fold_idx < len(test_performances[config_num]):
                if test_performances[config_num][fold_idx] is not None:
                    last_test_value = test_performances[config_num][fold_idx]
        
        # Fill missing configs with last incumbent values
        for config_num in range(max_config + 1):
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


def calculate_mean_std_error(performances: Dict[int, List[float]]) -> Tuple[List[int], List[float], List[float]]:
    """
    Calculate mean and standard error (SEM) for each config number.
    
    Standard Error = std / sqrt(n), where n is the number of samples.
    
    Args:
        performances: Dict mapping config_number -> list of performances
        
    Returns:
        Tuple of (config_numbers, means, std_errors) as sorted lists
    """
    config_numbers = sorted(performances.keys())
    means = []
    std_errors = []
    
    for config_num in config_numbers:
        perfs = performances[config_num]
        n = len(perfs)
        mean = np.mean(perfs)
        std = np.std(perfs)
        # Standard Error of the Mean (SEM) = std / sqrt(n)
        std_error = std / np.sqrt(n) if n > 1 else 0.0
        means.append(mean)
        std_errors.append(std_error)
    
    return config_numbers, means, std_errors


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
    extend_to_max_configs: bool = False,
    use_standard_error: bool = True
):
    """
    Create plots for validation and test performance over time.
    
    Args:
        experiment_dirs: List of paths to experiment directories
        all_validation_performances: List of tuples (experiment_name, validation_performances dict)
        all_test_performances: List of tuples (experiment_name, test_performances dict)
        output_path: Optional path to save the plot. If None, saves to first seed directory.
        extend_to_max_configs: Whether to extend shorter experiments to match the longest one
        use_standard_error: If True, use standard error (SEM) instead of standard deviation for error bars
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
    
    # Choose the appropriate calculation function based on use_standard_error flag
    calc_func = calculate_mean_std_error if use_standard_error else calculate_mean_std
    error_label = "standard error" if use_standard_error else "std"
    
    # Calculate global y-axis range across all plots for better comparison
    all_y_values = []
    for exp_name, validation_performances in all_validation_performances:
        if validation_performances:
            val_configs, val_means, val_errors = calc_func(validation_performances)
            if val_configs:
                # Debug: Print loaded validation values
                print(f"  Validation values for {exp_name}: means={val_means}, {error_label}s={val_errors}")
                # Only include reasonable values (AUC should be between 0 and 100)
                val_means_array = np.array(val_means)
                val_errors_array = np.array(val_errors)
                val_upper = val_means_array + val_errors_array
                val_lower = val_means_array - val_errors_array
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
            test_configs, test_means, test_errors = calc_func(test_performances)
            if test_configs:
                # Debug: Print loaded test values
                print(f"  Test values for {exp_name}: means={test_means}, {error_label}s={test_errors}")
                # Only include reasonable values (AUC should be between 0 and 100)
                test_means_array = np.array(test_means)
                test_errors_array = np.array(test_errors)
                test_upper = test_means_array + test_errors_array
                test_lower = test_means_array - test_errors_array
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
            val_configs, val_means, val_errors = calc_func(validation_performances)
            if val_configs:
                has_validation_data = True
                # Use different shades of blue for different experiments
                color = experiment_colors[idx] if len(experiment_dirs) > 1 else validation_color_base
                ax1.plot(val_configs, val_means, marker="o", linewidth=2, markersize=8, 
                        label=f"{exp_name} (Mean with {error_label})", color=color, linestyle="--")
                ax1.fill_between(
                    val_configs,
                    np.array(val_means) - np.array(val_errors),
                    np.array(val_means) + np.array(val_errors),
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
            test_configs, test_means, test_errors = calc_func(test_performances)
            if test_configs:
                has_test_data = True
                # Use different shades of green for different experiments
                color = experiment_colors[idx] if len(experiment_dirs) > 1 else test_color_base
                ax2.plot(test_configs, test_means, marker="s", linewidth=2, markersize=8, 
                        label=f"{exp_name} (Mean with {error_label})", color=color, linestyle="-")
                ax2.fill_between(
                    test_configs,
                    np.array(test_means) - np.array(test_errors),
                    np.array(test_means) + np.array(test_errors),
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
                val_configs, val_means, val_errors = calc_func(validation_performances)
                if val_configs:
                    color = experiment_colors[idx] if len(experiment_dirs) > 1 else validation_color_base
                    ax3.plot(val_configs, val_means, marker="o", linewidth=2, markersize=8, 
                            label=f"{exp_name} (Validation)", color=color, linestyle="--")
                    ax3.fill_between(
                        val_configs,
                        np.array(val_means) - np.array(val_errors),
                        np.array(val_means) + np.array(val_errors),
                        alpha=0.15,
                        color=color
                    )
        
        # Plot test for each experiment
        for idx, (exp_name, test_performances) in enumerate(all_test_performances):
            if test_performances:
                test_configs, test_means, test_errors = calc_func(test_performances)
                if test_configs:
                    color = experiment_colors[idx] if len(experiment_dirs) > 1 else test_color_base
                    ax3.plot(test_configs, test_means, marker="s", linewidth=2, markersize=8, 
                            label=f"{exp_name} (Test)", color=color, linestyle="-")
                    ax3.fill_between(
                        test_configs,
                        np.array(test_means) - np.array(test_errors),
                        np.array(test_means) + np.array(test_errors),
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
            output_path = seed_dirs[0] / "performance_over_configs.png"
        else:
            # Fallback to experiment directory if no seed directory found
            output_path = first_experiment_dir / "performance_over_configs.png"
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


def collect_performances_and_times_quicktune(
    experiment_dir: Path, 
    metric: str = "auc"
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Collect validation and test performances along with time information for QuickTune experiments.
    
    Args:
        experiment_dir: Path to experiment directory
        metric: Metric to use for validation performance (default: "auc")
        
    Returns:
        Tuple of (validation_performances, test_performances, time_points)
        Each is a dict mapping config_number -> list of values across outer folds
    """
    validation_performances_per_fold: Dict[int, List[Tuple[int, float]]] = {}
    test_performances_per_fold: Dict[int, List[Tuple[int, float]]] = {}
    times_per_fold: Dict[int, List[Tuple[int, float]]] = {}
    
    seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    
    if not seed_dirs:
        raise ValueError(f"No seed directories found in {experiment_dir}")
    
    print(f"Found {len(seed_dirs)} seed directory/ies")
    
    fold_counter = 0
    
    for seed_dir in seed_dirs:
        outer_fold_dirs = sorted(
            [d for d in seed_dir.iterdir() 
             if d.is_dir() and d.name.startswith("cv_outer_fold_")]
        )
        
        if not outer_fold_dirs:
            print(f"Warning: No outer fold directories found in {seed_dir}, skipping...")
            continue
        
        print(f"  Processing {len(outer_fold_dirs)} outer fold(s) in {seed_dir.name}")
        
        for outer_fold_dir in outer_fold_dirs:
            tuner_dir = outer_fold_dir / "tuner"
            
            if not tuner_dir.exists():
                print(f"Warning: tuner directory not found in {outer_fold_dir}, skipping...")
                continue
            
            config_dirs = []
            for item in tuner_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    try:
                        config_num = int(item.name)
                        config_dirs.append((config_num, item))
                    except ValueError:
                        continue
            
            config_dirs = sorted(config_dirs, key=lambda x: x[0])
            
            if not config_dirs:
                print(f"Warning: No config directories found in {tuner_dir}, skipping...")
                continue
            
            validation_performances_per_fold[fold_counter] = []
            test_performances_per_fold[fold_counter] = []
            times_per_fold[fold_counter] = []
            
            cumulative_time = 0.0
            
            for config_num, config_dir in config_dirs:
                # Load time from pipeline_result.json
                pipeline_result_path = config_dir / "pipeline_result.json"
                config_time = None
                
                if pipeline_result_path.exists():
                    try:
                        with open(pipeline_result_path, "r") as f:
                            pipeline_result = json.load(f)
                        cost = pipeline_result.get("cost", None)
                        if cost is not None:
                            cumulative_time += float(cost)  # cost is in seconds
                            config_time = cumulative_time
                    except Exception as e:
                        print(f"Warning: Could not load time from {pipeline_result_path}: {e}")
                
                if config_time is None:
                    continue
                
                # Load validation and test performance
                test_results_path = config_dir / "test_evaluation_results.json"
                if not test_results_path.exists():
                    continue
                
                val_perf = None
                try:
                    val_perf = load_validation_performance_quicktune(test_results_path, metric)
                except Exception as e:
                    continue
                
                if val_perf is not None:
                    validation_performances_per_fold[fold_counter].append((config_num, val_perf))
                    times_per_fold[fold_counter].append((config_num, config_time))
                
                try:
                    test_perf = load_test_performance(test_results_path)
                    test_performances_per_fold[fold_counter].append((config_num, test_perf))
                except Exception:
                    pass
            
            fold_counter += 1
    
    # Calculate incumbent performances and times for each fold independently
    # Structure: time_point -> (val_incumbent, test_perf_of_best_val_config) per fold
    # We need to collect all time points and their corresponding incumbent values
    all_time_points_per_fold: Dict[int, List[Tuple[float, float, float]]] = {}  # fold_idx -> [(time, val_incumbent, test_perf), ...]
    
    for fold_idx in sorted(validation_performances_per_fold.keys()):
        val_perfs = validation_performances_per_fold.get(fold_idx, [])
        test_perfs = test_performances_per_fold.get(fold_idx, [])
        times = times_per_fold.get(fold_idx, [])
        
        if not val_perfs or not times:
            continue
        
        # Create combined list: (config_num, time, val_perf, test_perf)
        combined = []
        test_perfs_dict = dict(test_perfs) if test_perfs else {}
        times_dict = dict(times)
        val_perfs_dict = dict(val_perfs)
        
        for config_num in sorted(val_perfs_dict.keys()):
            if config_num in times_dict and config_num in val_perfs_dict:
                test_perf = test_perfs_dict.get(config_num, None)
                combined.append((config_num, times_dict[config_num], val_perfs_dict[config_num], test_perf))
        
        # Sort by time (not config number!)
        combined_sorted = sorted(combined, key=lambda x: x[1])
        
        # Calculate incumbent for each time point
        best_val_so_far = float('-inf')
        best_val_config = None
        time_incumbent_pairs = []
        
        for config_num, time, val_perf, test_perf in combined_sorted:
            # Update incumbent if this config has better validation performance
            if val_perf > best_val_so_far:
                best_val_so_far = val_perf
                best_val_config = config_num
            
            # Use test performance of the config with best validation so far
            test_perf_to_use = test_perfs_dict.get(best_val_config, None) if best_val_config is not None else None
            
            time_incumbent_pairs.append((time, best_val_so_far, test_perf_to_use))
        
        all_time_points_per_fold[fold_idx] = time_incumbent_pairs
    
    # Now aggregate across folds: for each unique time point, collect incumbent values
    # We need to create time-based bins or use the actual time points
    # Strategy: collect all unique time points, then for each time point, get the incumbent value
    # from each fold (using the last incumbent value up to that time)
    
    # Get all unique time points across all folds
    all_unique_times = set()
    for fold_data in all_time_points_per_fold.values():
        for time, _, _ in fold_data:
            all_unique_times.add(time)
    
    all_unique_times = sorted(all_unique_times)
    
    # For each time point, collect incumbent values from each fold
    # IMPORTANT: Only include time points where ALL folds have data
    # This ensures the mean incumbent performance only increases over time
    validation_performances: Dict[float, List[float]] = {}  # time -> [val_incumbent per fold]
    test_performances: Dict[float, List[float]] = {}  # time -> [test_perf per fold]
    
    num_folds = len(all_time_points_per_fold)
    
    for time_point in all_unique_times:
        validation_performances[time_point] = []
        test_performances[time_point] = []
        
        for fold_idx in sorted(all_time_points_per_fold.keys()):
            fold_data = all_time_points_per_fold[fold_idx]
            
            # Find the last incumbent value up to this time point
            last_val_incumbent = None
            last_test_perf = None
            
            for t, val_inc, test_perf in fold_data:
                if t <= time_point:
                    last_val_incumbent = val_inc
                    last_test_perf = test_perf
                else:
                    break
            
            if last_val_incumbent is not None:
                validation_performances[time_point].append(last_val_incumbent)
                if last_test_perf is not None:
                    test_performances[time_point].append(last_test_perf)
        
        # Only keep time points where ALL folds have data
        if len(validation_performances[time_point]) < num_folds:
            # Remove this time point if not all folds have data
            del validation_performances[time_point]
            if time_point in test_performances:
                del test_performances[time_point]
    
    # Convert to config-number-based structure for compatibility with plotting function
    # We'll use time points as "config numbers" (but they're actually time points)
    # Or better: create a mapping from sequential index to time point
    # Only include time points that are still in validation_performances (i.e., all folds have data)
    valid_time_points = sorted([t for t in all_unique_times if t in validation_performances])
    
    validation_performances_by_index: Dict[int, List[float]] = {}
    test_performances_by_index: Dict[int, List[float]] = {}
    time_points_by_index: Dict[int, List[float]] = {}
    
    for idx, time_point in enumerate(valid_time_points):
        validation_performances_by_index[idx] = validation_performances[time_point]
        if time_point in test_performances and test_performances[time_point]:
            test_performances_by_index[idx] = test_performances[time_point]
        time_points_by_index[idx] = [time_point] * len(validation_performances[time_point])
    
    return validation_performances_by_index, test_performances_by_index, time_points_by_index


def create_performance_over_time_plot_quicktune(
    experiment_dir: Path,
    output_path: Path = None,
    use_standard_error: bool = True,
    metric: str = "auc"
):
    """
    Create plots for validation and test performance over time (wall-clock time) for QuickTune experiments.
    
    Args:
        experiment_dir: Path to experiment directory
        output_path: Optional path to save the plot
        use_standard_error: If True, use standard error instead of std
        metric: Metric to use for validation performance (default: "auc")
    """
    seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        print(f"Warning: No seed directories found in {experiment_dir}")
        return
    
    seed_dir = seed_dirs[0]
    
    # Collect performances and times
    validation_performances, test_performances, time_points = collect_performances_and_times_quicktune(
        experiment_dir, metric=metric
    )
    
    if not validation_performances or not time_points:
        print("Warning: No performance or time data found")
        return
    
    # Determine time unit based on max time
    all_times = []
    for times in time_points.values():
        all_times.extend(times)
    
    if not all_times:
        print("Warning: No time data found")
        return
    
    max_time = max(all_times)
    
    if max_time >= 3600:
        time_unit = "hours"
        time_unit_label = "Time (hours)"
        time_conversion = 1.0 / 3600.0
    elif max_time >= 60:
        time_unit = "min"
        time_unit_label = "Time (minutes)"
        time_conversion = 1.0 / 60.0
    else:
        time_unit = "sec"
        time_unit_label = "Time (seconds)"
        time_conversion = 1.0
    
    # Prepare data for plotting
    # time_points is now indexed by sequential index (0, 1, 2, ...) where each index corresponds to a time point
    sorted_indices = sorted(time_points.keys())
    plot_time_points = []
    val_means = []
    val_errors = []
    test_means = []
    test_errors = []
    error_label = "standard error" if use_standard_error else "std"
    
    for idx in sorted_indices:
        # Get the time point (all values in the list should be the same)
        times = time_points[idx]
        if times:
            time_point = times[0] * time_conversion  # All times in the list are the same
            plot_time_points.append(time_point)
        else:
            continue
        
        val_perfs = validation_performances.get(idx, [])
        if val_perfs:
            val_means.append(np.mean(val_perfs))
            if use_standard_error:
                n = len(val_perfs)
                std = np.std(val_perfs) if n > 1 else 0.0
                val_errors.append(std / np.sqrt(n) if n > 1 else 0.0)
            else:
                val_errors.append(np.std(val_perfs) if len(val_perfs) > 1 else 0.0)
        else:
            val_means.append(0.0)
            val_errors.append(0.0)
        
        test_perfs = test_performances.get(idx, [])
        if test_perfs:
            test_means.append(np.mean(test_perfs))
            if use_standard_error:
                n = len(test_perfs)
                std = np.std(test_perfs) if n > 1 else 0.0
                test_errors.append(std / np.sqrt(n) if n > 1 else 0.0)
            else:
                test_errors.append(np.std(test_perfs) if len(test_perfs) > 1 else 0.0)
        else:
            test_means.append(0.0)
            test_errors.append(0.0)
    
    # Create plots (same structure as create_performance_over_time_plot)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    
    validation_color = "blue"
    test_color = "green"
    
    # Calculate y-axis range
    all_y_values = []
    if val_means:
        all_y_values.extend([m + e for m, e in zip(val_means, val_errors)])
        all_y_values.extend([m - e for m, e in zip(val_means, val_errors)])
    if test_means:
        all_y_values.extend([m + e for m, e in zip(test_means, test_errors)])
        all_y_values.extend([m - e for m, e in zip(test_means, test_errors)])
    
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.1
        y_lim_min = max(0, y_min - y_padding) if y_min < 50 else 50
        y_lim_max = 100
    else:
        y_lim_min = 50
        y_lim_max = 100
    
    # Plot validation performance
    if plot_time_points and val_means:
        ax1.plot(plot_time_points, val_means, marker="o", linewidth=2, markersize=8, 
                label=f"Validation (Mean with {error_label})", color=validation_color, linestyle="--")
        ax1.fill_between(
            plot_time_points,
            np.array(val_means) - np.array(val_errors),
            np.array(val_means) + np.array(val_errors),
            alpha=0.2,
            color=validation_color
        )
        ax1.set_xlabel(time_unit_label, fontsize=12)
        ax1.set_ylabel("Validation AUC", fontsize=12)
        ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax1.text(0.5, 0.5, "No validation data available", 
                ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
        ax1.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Plot test performance
    if plot_time_points and test_means:
        ax2.plot(plot_time_points, test_means, marker="s", linewidth=2, markersize=8, 
                label=f"Test (Mean with {error_label})", color=test_color, linestyle="-")
        ax2.fill_between(
            plot_time_points,
            np.array(test_means) - np.array(test_errors),
            np.array(test_means) + np.array(test_errors),
            alpha=0.2,
            color=test_color
        )
        ax2.set_xlabel(time_unit_label, fontsize=12)
        ax2.set_ylabel("Test AUC", fontsize=12)
        ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax2.text(0.5, 0.5, "No test data available", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
        ax2.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Plot both together
    if (plot_time_points and val_means) or (plot_time_points and test_means):
        if plot_time_points and val_means:
            ax3.plot(plot_time_points, val_means, marker="o", linewidth=2, markersize=8, 
                    label="Validation", color=validation_color, linestyle="--")
            ax3.fill_between(
                plot_time_points,
                np.array(val_means) - np.array(val_errors),
                np.array(val_means) + np.array(val_errors),
                alpha=0.15,
                color=validation_color
            )
        
        if plot_time_points and test_means:
            ax3.plot(plot_time_points, test_means, marker="s", linewidth=2, markersize=8, 
                    label="Test", color=test_color, linestyle="-")
            ax3.fill_between(
                plot_time_points,
                np.array(test_means) - np.array(test_errors),
                np.array(test_means) + np.array(test_errors),
                alpha=0.15,
                color=test_color
            )
        
        ax3.set_xlabel(time_unit_label, fontsize=12)
        ax3.set_ylabel("AUC", fontsize=12)
        ax3.set_title("Validation & Test Performance Over Time", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=9, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=0)
        ax3.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax3.text(0.5, 0.5, "No data available", 
                ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Validation & Test Performance Over Time", fontsize=14, fontweight="bold")
        ax3.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Add experiment name as suptitle
    experiment_name = get_experiment_name_with_prefix(experiment_dir)
    fig.suptitle(f"Performance Over Time: {experiment_name}", fontsize=16, fontweight="bold", y=1.02)
    
    # Determine output path
    if output_path is None:
        output_path = seed_dir / "performance_over_time.png"
    else:
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply tight layout
    plt.tight_layout()
    
    # Re-apply y-axis limits after tight_layout
    for ax in [ax1, ax2, ax3]:
        ax.set_autoscale_on(False)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    for ax in [ax1, ax2, ax3]:
        ax.set_autoscale_on(False)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plot saved to: {pdf_path}")
    
    plt.close()


def create_performance_over_time_plot(
    experiment_dir: Path,
    cost_to_spend: float,
    output_path: Path = None,
    use_standard_error: bool = True
):
    """
    Create plots for validation and test performance over time (wall-clock time).
    
    This function reads costs from costs_in_min.csv or costs_in_hours.csv and
    performances from incumbent_performances.csv to create plots showing
    performance over wall-clock time instead of number of configs.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., experiments/NePS/lipo/test_plotting_script)
        cost_to_spend: Total time budget in seconds (from experimental_setting.cost_to_spend)
        output_path: Optional path to save the plot. If None, saves to seed directory.
    """
    # Find seed directory
    seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        print(f"Warning: No seed directories found in {experiment_dir}")
        return
    
    seed_dir = seed_dirs[0]
    neps_output_dir = seed_dir / "NePS_output"
    
    if not neps_output_dir.exists():
        print(f"Warning: NePS_output directory not found: {neps_output_dir}")
        return
    
    # Determine time unit based on cost_to_spend
    if cost_to_spend >= 60 and cost_to_spend < 3600:
        time_unit = "min"
        time_unit_label = "Time (minutes)"
        costs_csv_path = neps_output_dir / "costs_in_min.csv"
    elif cost_to_spend >= 3600:
        time_unit = "hours"
        time_unit_label = "Time (hours)"
        costs_csv_path = neps_output_dir / "costs_in_hours.csv"
    else:
        # Default to seconds if cost_to_spend < 60
        time_unit = "sec"
        time_unit_label = "Time (seconds)"
        costs_csv_path = neps_output_dir / "costs_in_sec.csv"
    
    # Read costs CSV - create it if it doesn't exist
    if not costs_csv_path.exists():
        print(f"Cost CSV file not found: {costs_csv_path}")
        print(f"Attempting to create cost CSV files from report.yaml files...")
        import sys
        from pathlib import Path
        # Add project root to path if not already there
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.utils.logging_utils import update_cost_csv_from_neps_output
        update_cost_csv_from_neps_output(str(neps_output_dir))
        
        # Check again if the file exists now
        if not costs_csv_path.exists():
            print(f"Warning: Could not create cost CSV file: {costs_csv_path}")
            print(f"Please ensure report.yaml files exist in config directories.")
            return
        else:
            print(f"Successfully created cost CSV files.")
    
    costs_df = pd.read_csv(costs_csv_path)
    
    # Read incumbent performances CSV
    performances_csv_path = neps_output_dir / "incumbent_performances.csv"
    if not performances_csv_path.exists():
        print(f"Warning: Incumbent performances CSV file not found: {performances_csv_path}")
        return
    
    performances_df = pd.read_csv(performances_csv_path)
    
    # Get all fold numbers from costs CSV
    all_folds = sorted(costs_df["outer_fold"].unique())
    
    # Calculate cumulative times per fold and per config
    # Structure: fold_num -> {config_num: cumulative_time}
    fold_cumulative_times = {}
    
    for fold_num in all_folds:
        fold_costs = costs_df[costs_df["outer_fold"] == fold_num].sort_values("config")
        cumulative_time = 0.0
        
        for _, cost_row in fold_costs.iterrows():
            config_num = int(cost_row["config"])
            evaluation_duration = cost_row["evaluation_duration"]
            
            # Add evaluation duration to cumulative time
            cumulative_time += evaluation_duration
            
            if fold_num not in fold_cumulative_times:
                fold_cumulative_times[fold_num] = {}
            fold_cumulative_times[fold_num][config_num] = cumulative_time
    
    # Determine time intervals based on time unit
    if time_unit == "hours":
        # For hours: check every hour from 1 to 24
        time_intervals = list(range(1, 25))  # 1, 2, ..., 24 hours
    elif time_unit == "min":
        # For minutes: check every minute (or every 5 minutes for efficiency)
        max_time_minutes = int(cost_to_spend / 60) + 1
        time_intervals = list(range(0, max_time_minutes + 1, 5))  # Every 5 minutes
    else:  # seconds
        # For seconds: check every 60 seconds (every minute)
        max_time_seconds = int(cost_to_spend) + 1
        time_intervals = list(range(0, max_time_seconds + 1, 60))  # Every 60 seconds
    
    # For each time interval, find the best config (best mean validation performance)
    # that was evaluated by that time across all folds
    time_points = []
    val_means = []
    val_errors = []
    test_means = []
    test_errors = []
    error_label = "standard error" if use_standard_error else "std"
    
    for time_interval in time_intervals:
        # Find all configs that were evaluated by this time across all folds
        # A config is only considered "evaluated" if it was evaluated in ALL folds by this time
        available_configs = set()
        
        # First, find all configs that were evaluated in at least one fold
        all_evaluated_configs = set()
        for fold_num in all_folds:
            if fold_num in fold_cumulative_times:
                for config_num, cumulative_time in fold_cumulative_times[fold_num].items():
                    if cumulative_time <= time_interval:
                        all_evaluated_configs.add(config_num)
        
        # Then, check which configs were evaluated in ALL folds
        for config_num in all_evaluated_configs:
            evaluated_in_all_folds = True
            for fold_num in all_folds:
                if fold_num not in fold_cumulative_times:
                    evaluated_in_all_folds = False
                    break
                if config_num not in fold_cumulative_times[fold_num]:
                    evaluated_in_all_folds = False
                    break
                if fold_cumulative_times[fold_num][config_num] > time_interval:
                    evaluated_in_all_folds = False
                    break
            
            if evaluated_in_all_folds:
                available_configs.add(config_num)
        
        if not available_configs:
            # No configs evaluated yet at this time: use 0 as performance
            time_points.append(time_interval)
            val_means.append(0.0)
            val_errors.append(0.0)
            test_means.append(0.0)
            test_errors.append(0.0)
            continue
        
        # For each available config, check if it has validation_mean in performances_df
        # Find the config with the best validation_mean
        best_val_mean = float('-inf')
        best_config = None
        best_test_mean = None
        
        for config_num in available_configs:
            perf_row = performances_df[performances_df["config"] == config_num]
            if not perf_row.empty:
                val_mean = perf_row["validation_mean"].iloc[0]
                if pd.notna(val_mean) and val_mean != "":
                    val_mean_float = float(val_mean)
                    if val_mean_float > best_val_mean:
                        best_val_mean = val_mean_float
                        best_config = config_num
                        test_mean = perf_row["test_mean"].iloc[0]
                        if pd.notna(test_mean) and test_mean != "":
                            best_test_mean = float(test_mean)
        
        if best_config is not None:
            # Get validation and test performances across all folds for this config
            perf_row = performances_df[performances_df["config"] == best_config]
            if not perf_row.empty:
                val_perfs = []
                test_perfs = []
                
                for fold_num in all_folds:
                    val_col = f"validation_fold_{fold_num}"
                    test_col = f"test_fold_{fold_num}"
                    
                    if val_col in perf_row.columns:
                        val_perf = perf_row[val_col].iloc[0]
                        if pd.notna(val_perf) and val_perf != "":
                            val_perfs.append(float(val_perf))
                    
                    if test_col in perf_row.columns:
                        test_perf = perf_row[test_col].iloc[0]
                        if pd.notna(test_perf) and test_perf != "":
                            test_perfs.append(float(test_perf))
                
                # Use validation_mean and test_mean from CSV (already averaged over folds)
                # But we still need to calculate standard error across folds
                if val_perfs:
                    val_mean = np.mean(val_perfs)  # Should match validation_mean from CSV
                    if use_standard_error:
                        n = len(val_perfs)
                        std = np.std(val_perfs) if n > 1 else 0.0
                        val_error = std / np.sqrt(n) if n > 1 else 0.0
                    else:
                        val_error = np.std(val_perfs) if len(val_perfs) > 1 else 0.0
                else:
                    val_mean = best_val_mean
                    val_error = 0.0
                
                if test_perfs:
                    test_mean = np.mean(test_perfs)  # Should match test_mean from CSV
                    if use_standard_error:
                        n = len(test_perfs)
                        std = np.std(test_perfs) if n > 1 else 0.0
                        test_error = std / np.sqrt(n) if n > 1 else 0.0
                    else:
                        test_error = np.std(test_perfs) if len(test_perfs) > 1 else 0.0
                else:
                    test_mean = best_test_mean if best_test_mean is not None else 0.0
                    test_error = 0.0
                
                time_points.append(time_interval)
                val_means.append(val_mean)
                val_errors.append(val_error)
                test_means.append(test_mean)
                test_errors.append(test_error)
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    
    validation_color = "blue"
    test_color = "green"
    
    # Calculate y-axis range
    all_y_values = []
    if val_means:
        all_y_values.extend([m + e for m, e in zip(val_means, val_errors)])
        all_y_values.extend([m - e for m, e in zip(val_means, val_errors)])
    if test_means:
        all_y_values.extend([m + e for m, e in zip(test_means, test_errors)])
        all_y_values.extend([m - e for m, e in zip(test_means, test_errors)])
    
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min
        y_padding = y_range * 0.1
        y_lim_min = max(0, y_min - y_padding) if y_min < 50 else 50
        y_lim_max = 100
    else:
        y_lim_min = 50
        y_lim_max = 100
    
    # Plot validation performance
    if time_points and val_means:
        ax1.plot(time_points, val_means, marker="o", linewidth=2, markersize=8, 
                label=f"Validation (Mean with {error_label})", color=validation_color, linestyle="-")
        ax1.fill_between(
            time_points,
            np.array(val_means) - np.array(val_errors),
            np.array(val_means) + np.array(val_errors),
            alpha=0.2,
            color=validation_color
        )
        ax1.set_xlabel(time_unit_label, fontsize=12)
        ax1.set_ylabel("Validation AUC", fontsize=12)
        ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3)
        if time_unit == "hours":
            ax1.set_xlim(left=1, right=24)
            ax1.set_xticks(range(1, 25))  # 1, 2, ..., 24
        else:
            ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax1.text(0.5, 0.5, "No validation data available", 
                ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
        if time_unit == "hours":
            ax1.set_xlim(left=1, right=24)
            ax1.set_xticks(range(1, 25))  # 1, 2, ..., 24
        else:
            ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Plot test performance
    if time_points and test_means:
        ax2.plot(time_points, test_means, marker="s", linewidth=2, markersize=8, 
                label=f"Test (Mean with {error_label})", color=test_color, linestyle="-")
        ax2.fill_between(
            time_points,
            np.array(test_means) - np.array(test_errors),
            np.array(test_means) + np.array(test_errors),
            alpha=0.2,
            color=test_color
        )
        ax2.set_xlabel(time_unit_label, fontsize=12)
        ax2.set_ylabel("Test AUC", fontsize=12)
        ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(True, alpha=0.3)
        if time_unit == "hours":
            ax2.set_xlim(left=1, right=24)
            ax2.set_xticks(range(1, 25))  # 1, 2, ..., 24
        else:
            ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax2.text(0.5, 0.5, "No test data available", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
        if time_unit == "hours":
            ax2.set_xlim(left=1, right=24)
            ax2.set_xticks(range(1, 25))  # 1, 2, ..., 24
        else:
            ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Plot both together
    if (time_points and val_means) or (time_points and test_means):
        if time_points and val_means:
            ax3.plot(time_points, val_means, marker="o", linewidth=2, markersize=8, 
                    label="Validation", color=validation_color, linestyle="--")
            ax3.fill_between(
                time_points,
                np.array(val_means) - np.array(val_errors),
                np.array(val_means) + np.array(val_errors),
                alpha=0.15,
                color=validation_color
            )
        
        if time_points and test_means:
            ax3.plot(time_points, test_means, marker="s", linewidth=2, markersize=8, 
                    label="Test", color=test_color, linestyle="-")
            ax3.fill_between(
                time_points,
                np.array(test_means) - np.array(test_errors),
                np.array(test_means) + np.array(test_errors),
                alpha=0.15,
                color=test_color
            )
        
        ax3.set_xlabel(time_unit_label, fontsize=12)
        ax3.set_ylabel("AUC", fontsize=12)
        ax3.set_title("Validation & Test Performance Over Time", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=9, loc='lower right')
        ax3.grid(True, alpha=0.3)
        if time_unit == "hours":
            ax3.set_xlim(left=1, right=24)
            ax3.set_xticks(range(1, 25))  # 1, 2, ..., 24
        else:
            ax3.set_xlim(left=0)
        ax3.set_ylim(bottom=y_lim_min, top=y_lim_max)
    else:
        ax3.text(0.5, 0.5, "No data available", 
                ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Validation & Test Performance Over Time", fontsize=14, fontweight="bold")
        if time_unit == "hours":
            ax3.set_xlim(left=1, right=24)
            ax3.set_xticks(range(1, 25))  # 1, 2, ..., 24
        else:
            ax3.set_xlim(left=0)
        ax3.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Add experiment name as suptitle
    experiment_name = get_experiment_name_with_prefix(experiment_dir)
    fig.suptitle(f"Performance Over Time: {experiment_name}", fontsize=16, fontweight="bold", y=1.02)
    
    # Determine output path
    if output_path is None:
        output_path = seed_dir / "performance_over_time.png"
    else:
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply tight layout
    plt.tight_layout()
    
    # Re-apply y-axis limits after tight_layout
    for ax in [ax1, ax2, ax3]:
        ax.set_autoscale_on(False)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    for ax in [ax1, ax2, ax3]:
        ax.set_autoscale_on(False)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plot saved to: {pdf_path}")
    
    plt.close()


def create_performance_over_time_plot_multi(
    experiment_dirs: List[Path],
    cost_to_spend: float,
    output_path: Path = None,
    use_standard_error: bool = True,
    y_min: float = None,
    y_max: float = None,
    title: str = None,
    normalize_to_baseline: bool = False
):
    """
    Create combined plots for multiple experiments showing performance over time (wall-clock time).
    
    For Baseline experiments (only 1 config), the performance is repeated for all 24 hours.
    For NePS experiments, the best validation performance at each hour is shown.
    
    Args:
        experiment_dirs: List of paths to experiment directories
        cost_to_spend: Total time budget in seconds
        output_path: Optional path to save the plot
        use_standard_error: If True, use standard error instead of std
        y_min: Optional minimum value for y-axis. If None, calculated from data.
        y_max: Optional maximum value for y-axis. If None, calculated from data.
        title: Optional title for the plot. If None, uses experiment names.
        normalize_to_baseline: If True, normalize all runs relative to baseline (baseline = 0%, y-axis: -10% to +10%)
    """
    # Determine time unit based on cost_to_spend
    if cost_to_spend >= 60 and cost_to_spend < 3600:
        time_unit = "min"
        time_unit_label = "Time (minutes)"
    elif cost_to_spend >= 3600:
        time_unit = "hours"
        time_unit_label = "Time (hours)"
    else:
        time_unit = "sec"
        time_unit_label = "Time (seconds)"
    
    # Determine time intervals
    if time_unit == "hours":
        max_time_hours = int(cost_to_spend / 3600)
        time_intervals = list(range(1, max_time_hours + 1))  # 1, 2, ..., max_time_hours
    elif time_unit == "min":
        max_time_minutes = int(cost_to_spend / 60) + 1
        time_intervals = list(range(0, max_time_minutes + 1, 5))
    else:
        max_time_seconds = int(cost_to_spend) + 1
        time_intervals = list(range(0, max_time_seconds + 1, 60))
    
    # Collect data for each experiment
    all_experiment_data = []
    
    for experiment_dir in experiment_dirs:
        exp_name = get_experiment_name_with_prefix(experiment_dir)
        print(f"\nProcessing experiment: {exp_name}")
        
        # Find seed directory
        seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        if not seed_dirs:
            print(f"Warning: No seed directories found in {experiment_dir}, skipping...")
            continue
        
        seed_dir = seed_dirs[0]
        
        # Detect experiment type: QuickTune has tuner directories, NePS/Baseline have NePS_output
        is_quicktune = False
        outer_fold_dirs = sorted([d for d in seed_dir.iterdir() if d.is_dir() and d.name.startswith("cv_outer_fold_")])
        if outer_fold_dirs:
            first_outer_fold = outer_fold_dirs[0]
            tuner_dir = first_outer_fold / "tuner"
            if tuner_dir.exists():
                is_quicktune = True
                print(f"Detected QuickTune experiment structure")
        
        if is_quicktune:
            # For QuickTune: use collect_performances_and_times_quicktune
            try:
                validation_performances, test_performances, time_points_dict = collect_performances_and_times_quicktune(
                    experiment_dir, metric="auc"
                )
                
                if not validation_performances or not time_points_dict:
                    print(f"Warning: No performance or time data found for QuickTune experiment, skipping...")
                    continue
                
                # Convert QuickTune time points to the same format as NePS/Baseline
                # time_points_dict maps index -> [time_point, time_point, ...] (one per fold)
                # We need to map these to the time_intervals (1-24 hours)
                
                # Get all unique time points and convert to hours
                all_unique_times = []
                for times_list in time_points_dict.values():
                    all_unique_times.extend(times_list)
                
                if not all_unique_times:
                    print(f"Warning: No time data found for QuickTune experiment, skipping...")
                    continue
                
                # Convert times to hours
                time_conversion = 1.0 / 3600.0 if time_unit == "hours" else (1.0 / 60.0 if time_unit == "min" else 1.0)
                
                # For each time interval, find the best validation performance up to that time
                # A config is only available if it was evaluated in ALL outer folds by this time
                time_points = []
                val_means = []
                val_errors = []
                test_means = []
                test_errors = []
                
                # Get number of folds from time_points_dict (each config should have same number of time points = number of folds)
                num_folds = 0
                if time_points_dict:
                    first_idx = next(iter(time_points_dict.keys()))
                    num_folds = len(time_points_dict[first_idx])
                
                for time_interval in time_intervals:
                    # Convert time_interval to seconds for comparison
                    if time_unit == "hours":
                        time_interval_sec = time_interval * 3600.0
                    elif time_unit == "min":
                        time_interval_sec = time_interval * 60.0
                    else:
                        time_interval_sec = time_interval
                    
                    # Find all configs (indices) that were evaluated in ALL folds by this time
                    available_indices = []
                    for idx in sorted(validation_performances.keys()):
                        times = time_points_dict.get(idx, [])
                        if len(times) == num_folds:  # Config must be evaluated in all folds
                            # Check if all fold times are <= time_interval_sec
                            if all(t <= time_interval_sec for t in times):
                                available_indices.append(idx)
                    
                    if not available_indices:
                        # No configs evaluated yet: use 0 as performance
                        time_points.append(time_interval)
                        val_means.append(0.0)
                        val_errors.append(0.0)
                        test_means.append(0.0)
                        test_errors.append(0.0)
                        continue
                    
                    # Find best config (best mean validation performance)
                    best_val_mean = float('-inf')
                    best_idx = None
                    
                    for idx in available_indices:
                        val_perfs = validation_performances.get(idx, [])
                        if val_perfs:
                            val_mean = np.mean(val_perfs)
                            if val_mean > best_val_mean:
                                best_val_mean = val_mean
                                best_idx = idx
                    
                    if best_idx is not None:
                        val_perfs = validation_performances.get(best_idx, [])
                        test_perfs = test_performances.get(best_idx, [])
                        
                        if val_perfs:
                            val_mean = np.mean(val_perfs)
                            if use_standard_error:
                                n = len(val_perfs)
                                std = np.std(val_perfs) if n > 1 else 0.0
                                val_error = std / np.sqrt(n) if n > 1 else 0.0
                            else:
                                val_error = np.std(val_perfs) if len(val_perfs) > 1 else 0.0
                        else:
                            val_mean = best_val_mean
                            val_error = 0.0
                        
                        if test_perfs:
                            test_mean = np.mean(test_perfs)
                            if use_standard_error:
                                n = len(test_perfs)
                                std = np.std(test_perfs) if n > 1 else 0.0
                                test_error = std / np.sqrt(n) if n > 1 else 0.0
                            else:
                                test_error = np.std(test_perfs) if len(test_perfs) > 1 else 0.0
                        else:
                            test_mean = 0.0
                            test_error = 0.0
                        
                        time_points.append(time_interval)
                        val_means.append(val_mean)
                        val_errors.append(val_error)
                        test_means.append(test_mean)
                        test_errors.append(test_error)
                
                all_experiment_data.append({
                    "name": exp_name,
                    "time_points": time_points,
                    "val_means": val_means,
                    "val_errors": val_errors,
                    "test_means": test_means,
                    "test_errors": test_errors
                })
                continue
                
            except Exception as e:
                print(f"Warning: Error processing QuickTune experiment: {e}, skipping...")
                continue
        
        # For NePS/Baseline: use existing logic
        neps_output_dir = seed_dir / "NePS_output"
        
        if not neps_output_dir.exists():
            print(f"Warning: NePS_output directory not found: {neps_output_dir}, skipping...")
            continue
        
        # Read incumbent performances CSV
        performances_csv_path = neps_output_dir / "incumbent_performances.csv"
        if not performances_csv_path.exists():
            print(f"Warning: Incumbent performances CSV file not found: {performances_csv_path}, skipping...")
            continue
        
        performances_df = pd.read_csv(performances_csv_path)
        
        # Check if this is a Baseline experiment (only 1 config)
        num_configs = len(performances_df)
        is_baseline = num_configs == 1
        
        if is_baseline:
            # For Baseline: check when the config was evaluated, show 0 before that time
            # Read costs CSV to determine evaluation time
            if time_unit == "hours":
                costs_csv_path = neps_output_dir / "costs_in_hours.csv"
            elif time_unit == "min":
                costs_csv_path = neps_output_dir / "costs_in_min.csv"
            else:
                costs_csv_path = neps_output_dir / "costs_in_sec.csv"
            
            if not costs_csv_path.exists():
                print(f"Warning: Cost CSV file not found: {costs_csv_path}")
                print(f"Attempting to create cost CSV files from report.yaml files...")
                import sys
                project_root = Path(__file__).parent.parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from src.utils.logging_utils import update_cost_csv_from_neps_output
                update_cost_csv_from_neps_output(str(neps_output_dir))
                
                if not costs_csv_path.exists():
                    print(f"Warning: Could not create cost CSV file: {costs_csv_path}, skipping...")
                    continue
            
            costs_df = pd.read_csv(costs_csv_path)
            all_folds = sorted(costs_df["outer_fold"].unique())
            
            # Calculate cumulative times per fold for config 1 (Baseline has only 1 config)
            fold_cumulative_times = {}
            for fold_num in all_folds:
                fold_costs = costs_df[costs_df["outer_fold"] == fold_num].sort_values("config")
                cumulative_time = 0.0
                
                for _, cost_row in fold_costs.iterrows():
                    config_num = int(cost_row["config"])
                    evaluation_duration = cost_row["evaluation_duration"]
                    cumulative_time += evaluation_duration
                    
                    if fold_num not in fold_cumulative_times:
                        fold_cumulative_times[fold_num] = {}
                    fold_cumulative_times[fold_num][config_num] = cumulative_time
            
            # Get the maximum time across all folds when config 1 was evaluated
            max_evaluation_time = 0.0
            for fold_num in all_folds:
                if fold_num in fold_cumulative_times and 1 in fold_cumulative_times[fold_num]:
                    max_evaluation_time = max(max_evaluation_time, fold_cumulative_times[fold_num][1])
            
            # Get performance data
            perf_row = performances_df.iloc[0]
            val_mean = float(perf_row["validation_mean"])
            test_mean = float(perf_row["test_mean"])
            
            # Get performances across folds for error calculation
            val_perfs = []
            test_perfs = []
            
            for fold_num in all_folds:
                val_col = f"validation_fold_{fold_num}"
                test_col = f"test_fold_{fold_num}"
                
                if val_col in perf_row and pd.notna(perf_row[val_col]):
                    val_perfs.append(float(perf_row[val_col]))
                if test_col in perf_row and pd.notna(perf_row[test_col]):
                    test_perfs.append(float(perf_row[test_col]))
            
            # Calculate errors
            if val_perfs:
                if use_standard_error:
                    n = len(val_perfs)
                    std = np.std(val_perfs) if n > 1 else 0.0
                    val_error = std / np.sqrt(n) if n > 1 else 0.0
                else:
                    val_error = np.std(val_perfs) if len(val_perfs) > 1 else 0.0
            else:
                val_error = 0.0
            
            if test_perfs:
                if use_standard_error:
                    n = len(test_perfs)
                    std = np.std(test_perfs) if n > 1 else 0.0
                    test_error = std / np.sqrt(n) if n > 1 else 0.0
                else:
                    test_error = np.std(test_perfs) if len(test_perfs) > 1 else 0.0
            else:
                test_error = 0.0
            
            # For each time interval, show 0 if before evaluation time, otherwise show performance
            time_points = []
            val_means = []
            val_errors = []
            test_means = []
            test_errors = []
            
            for time_interval in time_intervals:
                time_points.append(time_interval)
                
                if time_interval < max_evaluation_time:
                    # Before evaluation: show 0
                    val_means.append(0.0)
                    val_errors.append(0.0)
                    test_means.append(0.0)
                    test_errors.append(0.0)
                else:
                    # After evaluation: show actual performance
                    val_means.append(val_mean)
                    val_errors.append(val_error)
                    test_means.append(test_mean)
                    test_errors.append(test_error)
            
        else:
            # For NePS: use existing logic to find best config at each time interval
            # Read costs CSV
            if time_unit == "hours":
                costs_csv_path = neps_output_dir / "costs_in_hours.csv"
            elif time_unit == "min":
                costs_csv_path = neps_output_dir / "costs_in_min.csv"
            else:
                costs_csv_path = neps_output_dir / "costs_in_sec.csv"
            
            if not costs_csv_path.exists():
                print(f"Warning: Cost CSV file not found: {costs_csv_path}")
                print(f"Attempting to create cost CSV files from report.yaml files...")
                import sys
                project_root = Path(__file__).parent.parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from src.utils.logging_utils import update_cost_csv_from_neps_output
                update_cost_csv_from_neps_output(str(neps_output_dir))
                
                if not costs_csv_path.exists():
                    print(f"Warning: Could not create cost CSV file: {costs_csv_path}, skipping...")
                    continue
            
            costs_df = pd.read_csv(costs_csv_path)
            all_folds = sorted(costs_df["outer_fold"].unique())
            
            # Calculate cumulative times per fold
            fold_cumulative_times = {}
            for fold_num in all_folds:
                fold_costs = costs_df[costs_df["outer_fold"] == fold_num].sort_values("config")
                cumulative_time = 0.0
                
                for _, cost_row in fold_costs.iterrows():
                    config_num = int(cost_row["config"])
                    evaluation_duration = cost_row["evaluation_duration"]
                    cumulative_time += evaluation_duration
                    
                    if fold_num not in fold_cumulative_times:
                        fold_cumulative_times[fold_num] = {}
                    fold_cumulative_times[fold_num][config_num] = cumulative_time
            
            # For each time interval, find best config
            time_points = []
            val_means = []
            val_errors = []
            test_means = []
            test_errors = []
            
            for time_interval in time_intervals:
                # A config is only available if it was evaluated in ALL folds by this time
                available_configs = set()
                # First, find all configs that were evaluated in at least one fold
                all_evaluated_configs = set()
                for fold_num in all_folds:
                    if fold_num in fold_cumulative_times:
                        for config_num, cumulative_time in fold_cumulative_times[fold_num].items():
                            if cumulative_time <= time_interval:
                                all_evaluated_configs.add(config_num)
                
                # Then, check which configs were evaluated in ALL folds
                for config_num in all_evaluated_configs:
                    evaluated_in_all_folds = True
                    for fold_num in all_folds:
                        if fold_num not in fold_cumulative_times:
                            evaluated_in_all_folds = False
                            break
                        if config_num not in fold_cumulative_times[fold_num]:
                            evaluated_in_all_folds = False
                            break
                        if fold_cumulative_times[fold_num][config_num] > time_interval:
                            evaluated_in_all_folds = False
                            break
                    
                    if evaluated_in_all_folds:
                        available_configs.add(config_num)
                
                if not available_configs:
                    # No configs evaluated yet: use 0 as performance
                    time_points.append(time_interval)
                    val_means.append(0.0)
                    val_errors.append(0.0)
                    test_means.append(0.0)
                    test_errors.append(0.0)
                    continue
                
                # Find best config
                best_val_mean = float('-inf')
                best_config = None
                best_test_mean = None
                
                for config_num in available_configs:
                    perf_row = performances_df[performances_df["config"] == config_num]
                    if not perf_row.empty:
                        val_mean = perf_row["validation_mean"].iloc[0]
                        if pd.notna(val_mean) and val_mean != "":
                            val_mean_float = float(val_mean)
                            if val_mean_float > best_val_mean:
                                best_val_mean = val_mean_float
                                best_config = config_num
                                test_mean = perf_row["test_mean"].iloc[0]
                                if pd.notna(test_mean) and test_mean != "":
                                    best_test_mean = float(test_mean)
                
                if best_config is not None:
                    perf_row = performances_df[performances_df["config"] == best_config]
                    if not perf_row.empty:
                        val_perfs = []
                        test_perfs = []
                        
                        for fold_num in all_folds:
                            val_col = f"validation_fold_{fold_num}"
                            test_col = f"test_fold_{fold_num}"
                            
                            if val_col in perf_row.columns:
                                val_perf = perf_row[val_col].iloc[0]
                                if pd.notna(val_perf) and val_perf != "":
                                    val_perfs.append(float(val_perf))
                            
                            if test_col in perf_row.columns:
                                test_perf = perf_row[test_col].iloc[0]
                                if pd.notna(test_perf) and test_perf != "":
                                    test_perfs.append(float(test_perf))
                        
                        if val_perfs:
                            val_mean = np.mean(val_perfs)
                            if use_standard_error:
                                n = len(val_perfs)
                                std = np.std(val_perfs) if n > 1 else 0.0
                                val_error = std / np.sqrt(n) if n > 1 else 0.0
                            else:
                                val_error = np.std(val_perfs) if len(val_perfs) > 1 else 0.0
                        else:
                            val_mean = best_val_mean
                            val_error = 0.0
                        
                        if test_perfs:
                            test_mean = np.mean(test_perfs)
                            if use_standard_error:
                                n = len(test_perfs)
                                std = np.std(test_perfs) if n > 1 else 0.0
                                test_error = std / np.sqrt(n) if n > 1 else 0.0
                            else:
                                test_error = np.std(test_perfs) if len(test_perfs) > 1 else 0.0
                        else:
                            test_mean = best_test_mean if best_test_mean is not None else 0.0
                            test_error = 0.0
                        
                        time_points.append(time_interval)
                        val_means.append(val_mean)
                        val_errors.append(val_error)
                        test_means.append(test_mean)
                        test_errors.append(test_error)
        
        all_experiment_data.append({
            "name": exp_name,
            "time_points": time_points,
            "val_means": val_means,
            "val_errors": val_errors,
            "test_means": test_means,
            "test_errors": test_errors
        })
    
    if not all_experiment_data:
        print("Error: No valid experiment data found")
        return
    
    # Normalize to baseline if requested
    if normalize_to_baseline:
        # Find baseline experiment (first one with "baseline" in name, or first one if none found)
        baseline_idx = None
        for idx, exp_data in enumerate(all_experiment_data):
            if "baseline" in exp_data["name"].lower():
                baseline_idx = idx
                break
        
        # If no baseline found, use first experiment as baseline
        if baseline_idx is None:
            baseline_idx = 0
            print(f"Warning: No baseline experiment found, using first experiment as baseline: {all_experiment_data[0]['name']}")
        else:
            print(f"Using baseline: {all_experiment_data[baseline_idx]['name']}")
        
        baseline_data = all_experiment_data[baseline_idx]
        
        # Normalize all experiments relative to baseline
        # For each time point, subtract baseline value from experiment value
        # First, find the baseline evaluation time (when baseline switches from 0 to actual value)
        baseline_eval_time = None
        baseline_perf_value = None
        baseline_test_value = None
        
        if baseline_data["time_points"] and baseline_data["val_means"]:
            # Find when baseline switches from 0 to actual performance
            for time_point, val_mean in zip(baseline_data["time_points"], baseline_data["val_means"]):
                if val_mean > 0.0:  # First non-zero value indicates evaluation completed
                    baseline_eval_time = time_point
                    baseline_perf_value = val_mean
                    break
            
            # If no switch found, use the last non-zero value
            if baseline_eval_time is None:
                for time_point, val_mean in zip(reversed(baseline_data["time_points"]), reversed(baseline_data["val_means"])):
                    if val_mean > 0.0:
                        baseline_eval_time = time_point
                        baseline_perf_value = val_mean
                        break
        
        if baseline_data["time_points"] and baseline_data["test_means"]:
            # Find when baseline test switches from 0 to actual performance
            for time_point, test_mean in zip(baseline_data["time_points"], baseline_data["test_means"]):
                if test_mean > 0.0:
                    baseline_test_value = test_mean
                    break
            
            # If no switch found, use the last non-zero value
            if baseline_test_value is None:
                for time_point, test_mean in zip(reversed(baseline_data["time_points"]), reversed(baseline_data["test_means"])):
                    if test_mean > 0.0:
                        baseline_test_value = test_mean
                        break
        
        # If still no baseline values found, use the last values
        if baseline_perf_value is None and baseline_data["val_means"]:
            baseline_perf_value = baseline_data["val_means"][-1] if baseline_data["val_means"] else 0.0
        if baseline_test_value is None and baseline_data["test_means"]:
            baseline_test_value = baseline_data["test_means"][-1] if baseline_data["test_means"] else 0.0
        
        print(f"Baseline evaluation time: {baseline_eval_time}, baseline performance: {baseline_perf_value}, baseline test: {baseline_test_value}")
        
        for exp_data in all_experiment_data:
            # Normalize validation means and errors
            if exp_data["time_points"] and exp_data["val_means"]:
                normalized_val_means = []
                normalized_val_errors = []
                
                for time_point, val_mean, val_error in zip(exp_data["time_points"], exp_data["val_means"], exp_data["val_errors"]):
                    # Determine baseline value for this time point
                    if baseline_eval_time is not None and time_point < baseline_eval_time:
                        # Before baseline evaluation: use 0
                        baseline_val = 0.0
                    else:
                        # After baseline evaluation: use baseline performance value
                        baseline_val = baseline_perf_value if baseline_perf_value is not None else 0.0
                    
                    # Normalize: (value - baseline) in percentage points
                    normalized_val_means.append(val_mean - baseline_val)
                    # Error stays the same (absolute error, not relative)
                    normalized_val_errors.append(val_error)
                
                exp_data["val_means"] = normalized_val_means
                exp_data["val_errors"] = normalized_val_errors
            
            # Normalize test means and errors
            if exp_data["time_points"] and exp_data["test_means"]:
                normalized_test_means = []
                normalized_test_errors = []
                
                for time_point, test_mean, test_error in zip(exp_data["time_points"], exp_data["test_means"], exp_data["test_errors"]):
                    # Determine baseline value for this time point
                    if baseline_eval_time is not None and time_point < baseline_eval_time:
                        # Before baseline evaluation: use 0
                        baseline_test = 0.0
                    else:
                        # After baseline evaluation: use baseline test value
                        baseline_test = baseline_test_value if baseline_test_value is not None else 0.0
                    
                    # Normalize: (value - baseline) in percentage points
                    normalized_test_means.append(test_mean - baseline_test)
                    # Error stays the same (absolute error, not relative)
                    normalized_test_errors.append(test_error)
                
                exp_data["test_means"] = normalized_test_means
                exp_data["test_errors"] = normalized_test_errors
        
        # Baseline should be at 0 (already normalized)
        print(f"Normalized all experiments relative to baseline: {baseline_data['name']}")
    
    # Create combined plots with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors for different experiments based on experiment type and name
    # Baseline densenet: red, Baseline resnet: pink, NePS: orange, NePS bo: dark blue, NePS random-search: green, QuickTune: light blue
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    red = tab10_colors[3]
    pink = tab10_colors[6]
    orange = tab10_colors[1]
    dark_blue = tab10_colors[0]
    green = tab10_colors[2]
    light_blue = tab10_colors[9]
    purple = tab10_colors[4]
    brown = tab10_colors[5]
    
    def get_color_for_experiment(exp_name):
        """Get color based on experiment type and name."""
        exp_name_lower = exp_name.lower()
        if "baseline_" in exp_name_lower:
            if "densenet" in exp_name_lower:
                return red
            elif "resnet" in exp_name_lower:
                return pink
        elif "neps_" in exp_name_lower:
            if "bo_" in exp_name_lower:
                if "autonorm" in exp_name_lower:
                    return purple
                elif "learning-rate" in exp_name_lower:
                    return brown
                else:
                    return dark_blue
            elif "random-search_" in exp_name_lower:
                return green
            else:
                return orange
        elif "quicktune_" in exp_name_lower:
            return light_blue
        # Fallback: use index-based color
        return tab10_colors[0]
    
    experiment_colors = [get_color_for_experiment(exp_data["name"]) for exp_data in all_experiment_data]
    error_label = "standard error" if use_standard_error else "std"
    
    # Calculate y-axis range
    all_y_values = []
    for exp_data in all_experiment_data:
        if exp_data["val_means"]:
            all_y_values.extend([m + e for m, e in zip(exp_data["val_means"], exp_data["val_errors"])])
            all_y_values.extend([m - e for m, e in zip(exp_data["val_means"], exp_data["val_errors"])])
        if exp_data["test_means"]:
            all_y_values.extend([m + e for m, e in zip(exp_data["test_means"], exp_data["test_errors"])])
            all_y_values.extend([m - e for m, e in zip(exp_data["test_means"], exp_data["test_errors"])])
    
    # Use user-specified values if provided, otherwise calculate from data
    if normalize_to_baseline:
        # For normalized plots: y-axis from -10% to +10%
        if y_min is not None:
            y_lim_min = y_min
        else:
            y_lim_min = -10.0
        if y_max is not None:
            y_lim_max = y_max
        else:
            y_lim_max = 10.0
    else:
        # Original behavior
        if y_min is not None:
            y_lim_min = y_min
        elif all_y_values:
            data_y_min = min(all_y_values)
            y_range = max(all_y_values) - data_y_min
            y_padding = y_range * 0.1
            y_lim_min = max(0, data_y_min - y_padding) if data_y_min < 50 else 50
        else:
            y_lim_min = 50
        
        if y_max is not None:
            y_lim_max = y_max
        elif all_y_values:
            y_lim_max = 100
        else:
            y_lim_max = 100
    
    # Plot validation performance
    for idx, exp_data in enumerate(all_experiment_data):
        if exp_data["time_points"] and exp_data["val_means"]:
            color = experiment_colors[idx]
            ax1.plot(exp_data["time_points"], exp_data["val_means"], marker="o", linewidth=2, markersize=8,
                    label=exp_data['name'], color=color, linestyle="--")
            ax1.fill_between(
                exp_data["time_points"],
                np.array(exp_data["val_means"]) - np.array(exp_data["val_errors"]),
                np.array(exp_data["val_means"]) + np.array(exp_data["val_errors"]),
                alpha=0.2,
                color=color
            )
    
    # Add horizontal line at y=0 for baseline normalization
    if normalize_to_baseline:
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, label='Baseline (0%)')
    
    ax1.set_xlabel(time_unit_label, fontsize=12)
    if normalize_to_baseline:
        ax1.set_ylabel("Relative Improvement over Baseline (%)", fontsize=12)
    else:
        ax1.set_ylabel("Validation AUC", fontsize=12)
    ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    if time_unit == "hours":
        ax1.set_xlim(left=1, right=24)
        ax1.set_xticks(range(1, 25))
    else:
        ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Plot test performance
    for idx, exp_data in enumerate(all_experiment_data):
        if exp_data["time_points"] and exp_data["test_means"]:
            color = experiment_colors[idx]
            ax2.plot(exp_data["time_points"], exp_data["test_means"], marker="o", linewidth=2, markersize=8,
                    label=exp_data['name'], color=color, linestyle="-")
            ax2.fill_between(
                exp_data["time_points"],
                np.array(exp_data["test_means"]) - np.array(exp_data["test_errors"]),
                np.array(exp_data["test_means"]) + np.array(exp_data["test_errors"]),
                alpha=0.2,
                color=color
            )
    
    # Add horizontal line at y=0 for baseline normalization
    if normalize_to_baseline:
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3, label='Baseline (0%)')
    
    ax2.set_xlabel(time_unit_label, fontsize=12)
    if normalize_to_baseline:
        ax2.set_ylabel("Relative Improvement over Baseline (%)", fontsize=12)
    else:
        ax2.set_ylabel("Test AUC", fontsize=12)
    ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    if time_unit == "hours":
        max_time_hours = int(cost_to_spend / 3600)
        ax2.set_xlim(left=1, right=max_time_hours)
        ax2.set_xticks(range(1, max_time_hours + 1))
    else:
        ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    # Add suptitle
    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    else:
        experiment_names = ", ".join([exp_data["name"] for exp_data in all_experiment_data])
        fig.suptitle(f"Performance Over Time: {experiment_names}", fontsize=16, fontweight="bold", y=1.02)
    
    # Determine output path
    if output_path is None:
        output_dir = Path("experiments/Plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "performance_over_time_multi.png"
    else:
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply tight layout
    plt.tight_layout()
    
    # Re-apply y-axis limits after tight_layout
    for ax in [ax1, ax2]:
        ax.set_autoscale_on(False)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    for ax in [ax1, ax2]:
        ax.set_autoscale_on(False)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plot saved to: {pdf_path}")
    
    plt.close()


def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot test and validation performance over time for NePS and QuickTune experiments. "
                    "Can plot single or multiple experiments together. Automatically detects experiment type."
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
    parser.add_argument(
        "--over-time",
        action="store_true",
        help="If set, plot performance over wall-clock time instead of number of configs. "
             "For QuickTune experiments, uses cost from pipeline_result.json files. "
             "For NePS experiments, requires cost_to_spend parameter and cost CSV files."
    )
    parser.add_argument(
        "--cost-to-spend",
        type=float,
        default=None,
        help="Total time budget in seconds (required for NePS experiments when using --over-time). "
             "Used to determine time unit (seconds/minutes/hours)."
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Minimum value for y-axis. If not specified, calculated from data."
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Maximum value for y-axis. If not specified, calculated from data."
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the plot. If not specified, uses experiment names."
    )
    parser.add_argument(
        "--normalize-to-baseline",
        action="store_true",
        help="If set, normalize all runs relative to baseline (baseline = 0%, y-axis: -10% to +10%). "
             "The baseline is automatically identified as the first experiment with 'baseline' in its name, "
             "or the first experiment if none found."
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
    
    # If --over-time is set, use time-based plotting
    if args.over_time:
        # If multiple experiments, use combined plotting
        if len(experiment_dirs) > 1:
            if args.cost_to_spend is None:
                raise ValueError("--cost-to-spend is required for NePS experiments when using --over-time with multiple experiments")
            output_path = Path(args.output) if args.output else None
            create_performance_over_time_plot_multi(
                experiment_dirs,
                cost_to_spend=args.cost_to_spend,
                output_path=output_path,
                y_min=args.y_min,
                y_max=args.y_max,
                title=args.title,
                normalize_to_baseline=args.normalize_to_baseline
            )
            print("\nDone!")
            return
        
        # Single experiment: use existing logic
        for experiment_dir in experiment_dirs:
            print(f"\nProcessing experiment: {experiment_dir.name}")
            
            # Detect experiment type
            is_quicktune = False
            seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
            if seed_dirs:
                first_seed_dir = seed_dirs[0]
                outer_fold_dirs = sorted([d for d in first_seed_dir.iterdir() if d.is_dir() and d.name.startswith("cv_outer_fold_")])
                if outer_fold_dirs:
                    first_outer_fold = outer_fold_dirs[0]
                    tuner_dir = first_outer_fold / "tuner"
                    if tuner_dir.exists():
                        is_quicktune = True
                        print(f"Detected QuickTune experiment structure")
            
            output_path = Path(args.output) if args.output else None
            if is_quicktune:
                # QuickTune: use time-based plotting
                create_performance_over_time_plot_quicktune(
                    experiment_dir,
                    output_path=output_path,
                    metric="auc"
                )
            else:
                # NePS: use existing time-based plotting (requires cost_to_spend)
                if args.cost_to_spend is None:
                    raise ValueError("--cost-to-spend is required for NePS experiments when using --over-time")
                create_performance_over_time_plot(
                    experiment_dir,
                    cost_to_spend=args.cost_to_spend,
                    output_path=output_path
                )
        
        print("\nDone!")
        return
    
    # Otherwise, use config-based plotting (existing behavior)
    # Collect performances for each experiment
    all_validation_performances = []
    all_test_performances = []
    
    for experiment_dir in experiment_dirs:
        print(f"\nProcessing experiment: {experiment_dir.name}")
        
        # Detect experiment type: QuickTune has tuner directories, NePS has NePS_output directories
        is_quicktune = False
        seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        if seed_dirs:
            # Check first seed directory for QuickTune structure (tuner in cv_outer_fold_*)
            first_seed_dir = seed_dirs[0]
            outer_fold_dirs = sorted([d for d in first_seed_dir.iterdir() if d.is_dir() and d.name.startswith("cv_outer_fold_")])
            if outer_fold_dirs:
                first_outer_fold = outer_fold_dirs[0]
                tuner_dir = first_outer_fold / "tuner"
                if tuner_dir.exists():
                    is_quicktune = True
                    print(f"Detected QuickTune experiment structure")
        
        # Use appropriate collection function based on experiment type
        if is_quicktune:
            # Get metric from path or default to "auc"
            # QuickTune experiments typically use "auc" as the metric
            validation_performances, test_performances = collect_performances_quicktune(experiment_dir, metric="auc")
        else:
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

