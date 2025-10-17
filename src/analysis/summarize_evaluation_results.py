#!/usr/bin/env python3
"""
Script to summarize test results across all cross-validation folds for NePS experiments.
Specifically designed for Baseline mode experiments with single config_1.
"""

import os
import json
import argparse
import statistics
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from src.evaluate_trained_config import calculate_metrics_from_probabilities

def load_test_results(experiment_path: str, seed: str = "42") -> Dict[str, List[float]]:
    """
    Load test evaluation results from all cv_outer_fold directories.
    
    Args:
        experiment_path: Path to the experiment directory
        seed: Seed directory name (default: "42")
    
    Returns:
        Dictionary with metric names as keys and lists of values across folds
    """
    neps_output_path = Path(experiment_path) / f"seed_{seed}" / "NePS_output"
    
    if not neps_output_path.exists():
        raise FileNotFoundError(f"NePS output directory not found: {neps_output_path}")
    
    # Find all cv_outer_fold directories
    cv_fold_dirs = sorted([d for d in neps_output_path.iterdir() 
                          if d.is_dir() and d.name.startswith("cv_outer_fold_")])
    
    if not cv_fold_dirs:
        raise FileNotFoundError(f"No cv_outer_fold directories found in {neps_output_path}")
    
    print(f"Found {len(cv_fold_dirs)} cross-validation folds: {[d.name for d in cv_fold_dirs]}")
    
    # Load results from each fold
    all_metrics = {}
    
    for fold_dir in cv_fold_dirs:
        results_file = fold_dir / "configs" / "config_1" / "test_evaluation_results.json"
        
        if not results_file.exists():
            print(f"Warning: Results file not found: {results_file}")
            continue
        
        print(f"Loading results from: {results_file}")
        
        with open(results_file, 'r') as f:
            fold_results = json.load(f)
        
        # Extract ensemble metrics (for Baseline mode)
        ensemble_metrics = fold_results.get("ensemble", {})
        
        # Flatten nested metrics
        for metric_name, value in ensemble_metrics.items():
            if metric_name == "confusion_matrix":
                continue  # Skip confusion matrix for now
            elif metric_name == "per_class":
                # Handle per-class metrics
                for class_metric, values in value.items():
                    for i, val in enumerate(values):
                        key = f"per_class_{class_metric}_class_{i}"
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(val)
            else:
                # Regular metrics
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
    
    return all_metrics


def load_predictions_for_outer_ensemble(experiment_path: str, seed: str = "42") -> Dict[str, Any]:
    """
    Load raw predictions from all cv_outer_fold directories for outer fold ensemble.
    
    Args:
        experiment_path: Path to the experiment directory
        seed: Seed directory name (default: "42")
    
    Returns:
        Dictionary containing predictions data from all outer folds
    """
    neps_output_path = Path(experiment_path) / f"seed_{seed}" / "NePS_output"
    
    if not neps_output_path.exists():
        raise FileNotFoundError(f"NePS output directory not found: {neps_output_path}")
    
    # Find all cv_outer_fold directories
    cv_fold_dirs = sorted([d for d in neps_output_path.iterdir() 
                          if d.is_dir() and d.name.startswith("cv_outer_fold_")])
    
    if not cv_fold_dirs:
        raise FileNotFoundError(f"No cv_outer_fold directories found in {neps_output_path}")
    
    print(f"Found {len(cv_fold_dirs)} cross-validation folds for outer ensemble: {[d.name for d in cv_fold_dirs]}")
    
    # Load predictions from each fold
    all_predictions = []
    ground_truth = None
    num_classes = None
    framework = None
    
    for fold_dir in cv_fold_dirs:
        predictions_file = fold_dir / "configs" / "config_1" / "test_predictions_for_outer_ensemble.json"
        
        if not predictions_file.exists():
            print(f"Warning: Predictions file not found: {predictions_file}")
            continue
        
        print(f"Loading predictions from: {predictions_file}")
        
        with open(predictions_file, 'r') as f:
            fold_predictions = json.load(f)
        
        all_predictions.append(np.array(fold_predictions["probabilities"]))
        
        # Store ground truth and metadata from first fold
        if ground_truth is None:
            ground_truth = np.array(fold_predictions["ground_truth"])
            num_classes = fold_predictions["num_classes"]
            framework = fold_predictions["framework"]
    
    if not all_predictions:
        raise FileNotFoundError("No prediction files found for outer fold ensemble")
    
    return {
        "predictions": all_predictions,
        "ground_truth": ground_truth,
        "num_classes": num_classes,
        "framework": framework,
        "num_outer_folds": len(all_predictions)
    }


def create_outer_fold_ensemble(predictions_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Create outer fold ensemble by averaging predictions across outer folds.
    
    Args:
        predictions_data: Dictionary containing predictions from all outer folds
    
    Returns:
        Dictionary with ensemble metrics
    """
    all_predictions = predictions_data["predictions"]
    ground_truth = predictions_data["ground_truth"]
    num_classes = predictions_data["num_classes"]
    
    # Stack all fold predictions into a 3D array
    # shape: (num_outer_folds, num_samples, num_classes)
    stacked_predictions = np.stack(all_predictions, axis=0)
    
    # Average across outer folds to get ensemble predictions
    # shape: (num_samples, num_classes)
    ensemble_predictions = np.mean(stacked_predictions, axis=0)
    
    print(f"\n=== Creating Outer Fold Ensemble ===")
    print(f"Total outer folds: {len(all_predictions)}")
    print(f"Test samples: {len(ground_truth)}")
    print(f"Ensemble prediction shape: {ensemble_predictions.shape}")
    
    # Calculate ensemble metrics from ensemble predictions
    ensemble_metrics = calculate_metrics_from_probabilities(
        ensemble_predictions, ground_truth, num_classes
    )
    
    print(f"\n=== Outer Fold Ensemble Results ===")
    print(f"Accuracy: {ensemble_metrics['accuracy']:.2f}%")
    print(f"AUC (macro): {ensemble_metrics['auc_macro']:.2f}%")
    print(f"AUC (micro): {ensemble_metrics['auc_micro']:.2f}%")
    print(f"F1 (macro): {ensemble_metrics['f1_macro']:.2f}%")
    print(f"F1 (micro): {ensemble_metrics['f1_micro']:.2f}%")
    
    return ensemble_metrics


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate mean, std, median, and median absolute deviation for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "mad": 0.0}
    
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    median_val = statistics.median(values)
    
    # Calculate median absolute deviation
    deviations = [abs(x - median_val) for x in values]
    mad_val = statistics.median(deviations)
    
    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "median": float(median_val),
        "mad": float(mad_val)
    }


def format_statistics(stats: Dict[str, float], metric_name: str) -> str:
    """
    Format statistics for output.
    
    Args:
        stats: Dictionary with statistical measures
        metric_name: Name of the metric
    
    Returns:
        Formatted string
    """
    return (f"{metric_name:30s}: "
            f"mean={stats['mean']:8.4f}±{stats['std']:8.4f}, "
            f"median={stats['median']:8.4f}±{stats['mad']:8.4f}")


def summarize_experiment(experiment_path: str, seed: str = "42") -> str:
    """
    Summarize test results for a complete experiment.
    Now includes both individual fold results and outer fold ensemble results.
    
    Args:
        experiment_path: Path to the experiment directory
        seed: Seed directory name
    
    Returns:
        Formatted summary string
    """
    print(f"Summarizing experiment: {experiment_path}")
    
    # Load all metrics across folds (individual fold approach)
    all_metrics = load_test_results(experiment_path, seed)
    
    if not all_metrics:
        return "No metrics found!"
    
    # Calculate statistics for each metric
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"TEST RESULTS SUMMARY")
    summary_lines.append(f"Experiment: {Path(experiment_path).name}")
    summary_lines.append(f"Seed: {seed}")
    summary_lines.append(f"Number of CV folds: {len(list(all_metrics.values())[0])}")
    summary_lines.append("=" * 80)
    summary_lines.append("")


    # Individual fold results
    # ------------------------------------------------------------------------------------------------
    summary_lines.append("INDIVIDUAL OUTERFOLD RESULTS:")
    summary_lines.append("-" * 60)
    
    # Group metrics by category
    main_metrics = []
    per_class_metrics = {}
    
    for metric_name in sorted(all_metrics.keys()):
        if metric_name.startswith("per_class_"):
            parts = metric_name.split("_")
            class_metric = f"{parts[2]}_{parts[3]}"  # e.g., "precision_class_0"
            if class_metric not in per_class_metrics:
                per_class_metrics[class_metric] = []
            per_class_metrics[class_metric].append(metric_name)
        else:
            main_metrics.append(metric_name)
    
    # Main metrics
    summary_lines.append("MAIN METRICS:")
    summary_lines.append("-" * 40)
    for metric_name in main_metrics:
        stats = calculate_statistics(all_metrics[metric_name])
        summary_lines.append(format_statistics(stats, metric_name))
    
    summary_lines.append("")
    
    # Per-class metrics
    if per_class_metrics:
        summary_lines.append("PER-CLASS METRICS:")
        summary_lines.append("-" * 40)
        for class_metric, metric_names in per_class_metrics.items():
            summary_lines.append(f"\n{class_metric.replace('_', ' ').title()}:")
            for metric_name in metric_names:
                stats = calculate_statistics(all_metrics[metric_name])
                class_name = metric_name.split("_")[-1]  # e.g., "class_0" -> "0"
                display_name = f"  Class {class_name}"
                summary_lines.append(format_statistics(stats, display_name))
    
    summary_lines.append("")

    # Post-hoc: Outer fold ensemble results
    # ------------------------------------------------------------------------------------------------
    summary_lines.append("POST-HOC: OUTER FOLD ENSEMBLE RESULTS:")
    summary_lines.append("-" * 60)
    
    # Load predictions for outer fold ensemble
    predictions_data = load_predictions_for_outer_ensemble(experiment_path, seed)
    
    # Create outer fold ensemble
    ensemble_metrics = create_outer_fold_ensemble(predictions_data)
    
    # Add ensemble results to summary
    summary_lines.append("ENSEMBLE METRICS:")
    summary_lines.append("-" * 40)
    for metric_name, value in ensemble_metrics.items():
        if metric_name not in ["confusion_matrix", "per_class"]:
            summary_lines.append(f"{metric_name:30s}: {value:8.4f}")

    summary_lines.append("")

    # Compare individual vs ensemble for key metrics
    summary_lines.append("")
    summary_lines.append("COMPARISON: ENSEMBLE VS. INDIVIDUAL OUTER FOLD RESULTS:")
    summary_lines.append("-" * 60)

    key_metrics = ["accuracy", "auc_macro", "auc_micro", "f1_macro", "f1_micro"]
    for metric in key_metrics:
        if metric in all_metrics and metric in ensemble_metrics:
            ensemble_value = ensemble_metrics[metric]
            individual_mean = statistics.mean(all_metrics[metric])
            improvement = ensemble_value - individual_mean
            summary_lines.append(f"{metric:20s}: Ensemble={ensemble_value:6.2f}, Individual={individual_mean:6.2f}, Δ={improvement:+6.2f}")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    return "\n".join(summary_lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize NePS test results across CV folds")
    parser.add_argument("experiment_path", help="Path to the experiment directory")
    parser.add_argument("--seed", default="42", help="Seed directory name (default: 42)")
    parser.add_argument("--output", help="Output file path (default: experiment_path/evaluation_summary_across_outer_fols.txt)")
    
    args = parser.parse_args()
    
    # Validate experiment path
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        print(f"Error: Experiment path does not exist: {experiment_path}")
        return 1
    
    # Generate summary
    try:
        summary = summarize_experiment(str(experiment_path), args.seed)
        
        # Determine output file
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = experiment_path / "evaluation_summary_across_outer_fols.txt"
        
        # Write summary to file
        with open(output_file, 'w') as f:
            f.write(summary)
        
        print(f"\nSummary saved to: {output_file}")
        print("\n" + summary)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
