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
                report_path = config_dir / "report.yaml"
                if report_path.exists():
                    try:
                        val_perf = load_validation_performance(report_path)
                        validation_performances_per_fold[fold_counter].append((config_num, val_perf))
                    except Exception as e:
                        print(f"Warning: Could not load validation performance from {report_path}: {e}")
                
                # Load test performance
                test_results_path = config_dir / "test_evaluation_results.json"
                if test_results_path.exists():
                    try:
                        test_perf = load_test_performance(test_results_path)
                        test_performances_per_fold[fold_counter].append((config_num, test_perf))
                    except Exception as e:
                        print(f"Warning: Could not load test performance from {test_results_path}: {e}")
            
            fold_counter += 1
    
    # Calculate incumbent performances for each fold
    validation_performances: Dict[int, List[float]] = {}
    test_performances: Dict[int, List[float]] = {}
    
    # Process validation performances
    for fold_idx, config_perfs in validation_performances_per_fold.items():
        if not config_perfs:
            continue
        
        # Sort by config number
        config_perfs_sorted = sorted(config_perfs, key=lambda x: x[0])
        
        # Calculate incumbent (best so far) for each config
        best_so_far = float('-inf')
        for config_num, perf in config_perfs_sorted:
            best_so_far = max(best_so_far, perf)  # For AUC, higher is better
            if config_num not in validation_performances:
                validation_performances[config_num] = []
            validation_performances[config_num].append(best_so_far)
    
    # Process test performances
    for fold_idx, config_perfs in test_performances_per_fold.items():
        if not config_perfs:
            continue
        
        # Sort by config number
        config_perfs_sorted = sorted(config_perfs, key=lambda x: x[0])
        
        # Calculate incumbent (best so far) for each config
        best_so_far = float('-inf')
        for config_num, perf in config_perfs_sorted:
            best_so_far = max(best_so_far, perf)  # For AUC, higher is better
            if config_num not in test_performances:
                test_performances[config_num] = []
            test_performances[config_num].append(best_so_far)
    
    return validation_performances, test_performances


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


def create_plots(
    experiment_dir: Path,
    validation_performances: Dict[int, List[float]],
    test_performances: Dict[int, List[float]],
    output_path: Path = None
):
    """
    Create plots for validation and test performance over time.
    
    Args:
        experiment_dir: Path to experiment directory
        validation_performances: Dict mapping config_number -> list of validation performances
        test_performances: Dict mapping config_number -> list of test performances
        output_path: Optional path to save the plot. If None, saves to first seed directory.
    """
    # Calculate mean and std for validation
    val_configs, val_means, val_stds = calculate_mean_std(validation_performances)
    
    # Calculate mean and std for test
    test_configs, test_means, test_stds = calculate_mean_std(test_performances)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    
    # Define colors: validation = blue, test = green
    validation_color = "blue"
    test_color = "green"
    
    # Plot validation performance
    if val_configs:
        ax1.plot(val_configs, val_means, marker="o", linewidth=2, markersize=8, label="Mean", color=validation_color)
        ax1.fill_between(
            val_configs,
            np.array(val_means) - np.array(val_stds),
            np.array(val_means) + np.array(val_stds),
            alpha=0.3,
            label="±1 Std",
            color=validation_color
        )
        ax1.set_xlabel("Number of Configs", fontsize=12)
        ax1.set_ylabel("Validation AUC", fontsize=12)
        ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        # Ensure x-axis shows all config numbers, especially when there's only 1 config
        if len(val_configs) == 1:
            ax1.set_xlim(left=0, right=val_configs[0] + 1)
            ax1.set_xticks([val_configs[0]])
        else:
            ax1.set_xticks(val_configs)
    else:
        ax1.text(0.5, 0.5, "No validation data available", 
                ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Validation Performance Over Time", fontsize=14, fontweight="bold")
    
    # Plot test performance
    if test_configs:
        ax2.plot(test_configs, test_means, marker="s", linewidth=2, markersize=8, label="Mean", color=test_color)
        ax2.fill_between(
            test_configs,
            np.array(test_means) - np.array(test_stds),
            np.array(test_means) + np.array(test_stds),
            alpha=0.3,
            label="±1 Std",
            color=test_color
        )
        ax2.set_xlabel("Number of Configs", fontsize=12)
        ax2.set_ylabel("Test AUC", fontsize=12)
        ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        # Ensure x-axis shows all config numbers, especially when there's only 1 config
        if len(test_configs) == 1:
            ax2.set_xlim(left=0, right=test_configs[0] + 1)
            ax2.set_xticks([test_configs[0]])
        else:
            ax2.set_xticks(test_configs)
    else:
        ax2.text(0.5, 0.5, "No test data available", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Test Performance Over Time", fontsize=14, fontweight="bold")
    
    # Plot both validation and test performance together
    if val_configs or test_configs:
        if val_configs:
            ax3.plot(val_configs, val_means, marker="o", linewidth=2, markersize=8, label="Validation Mean", color=validation_color)
            ax3.fill_between(
                val_configs,
                np.array(val_means) - np.array(val_stds),
                np.array(val_means) + np.array(val_stds),
                alpha=0.2,
                label="Validation ±1 Std",
                color=validation_color
            )
        
        if test_configs:
            ax3.plot(test_configs, test_means, marker="s", linewidth=2, markersize=8, label="Test Mean", color=test_color)
            ax3.fill_between(
                test_configs,
                np.array(test_means) - np.array(test_stds),
                np.array(test_means) + np.array(test_stds),
                alpha=0.2,
                label="Test ±1 Std",
                color=test_color
            )
        
        ax3.set_xlabel("Number of Configs", fontsize=12)
        ax3.set_ylabel("AUC", fontsize=12)
        ax3.set_title("Validation & Test Performance Over Time", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=0)
        # Ensure x-axis shows all config numbers, especially when there's only 1 config
        all_configs = sorted(set(val_configs + test_configs)) if val_configs and test_configs else (val_configs or test_configs)
        if len(all_configs) == 1:
            ax3.set_xlim(left=0, right=all_configs[0] + 1)
            ax3.set_xticks([all_configs[0]])
        else:
            ax3.set_xticks(all_configs)
    else:
        ax3.text(0.5, 0.5, "No data available", 
                ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Validation & Test Performance Over Time", fontsize=14, fontweight="bold")
    
    # Add experiment name as suptitle
    experiment_name = experiment_dir.name
    fig.suptitle(f"Performance Over Time: {experiment_name}", fontsize=16, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        # Find first seed directory
        seed_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        if seed_dirs:
            output_path = seed_dirs[0] / "performance_over_time.png"
        else:
            # Fallback to experiment directory if no seed directory found
            output_path = experiment_dir / "performance_over_time.png"
    else:
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plot saved to: {pdf_path}")
    
    plt.close()


def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot test and validation performance over time for NePS experiments"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (e.g., experiments/NePS/lipo/test_plotting_script)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the plot. If not specified, saves to experiment_dir/performance_over_time.png"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")
    
    print(f"Processing experiment: {experiment_dir}")
    print("=" * 60)
    
    # Collect performances
    validation_performances, test_performances = collect_performances(experiment_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Validation performances: {len(validation_performances)} configs")
    print(f"  Test performances: {len(test_performances)} configs")
    
    if validation_performances:
        print(f"\n  Validation configs: {sorted(validation_performances.keys())}")
        for config_num in sorted(validation_performances.keys()):
            perfs = validation_performances[config_num]
            print(f"    Config {config_num}: {len(perfs)} fold(s), mean={np.mean(perfs):.2f}, std={np.std(perfs):.2f}")
    
    if test_performances:
        print(f"\n  Test configs: {sorted(test_performances.keys())}")
        for config_num in sorted(test_performances.keys()):
            perfs = test_performances[config_num]
            print(f"    Config {config_num}: {len(perfs)} fold(s), mean={np.mean(perfs):.2f}, std={np.std(perfs):.2f}")
    
    # Create plots
    output_path = Path(args.output) if args.output else None
    create_plots(experiment_dir, validation_performances, test_performances, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

