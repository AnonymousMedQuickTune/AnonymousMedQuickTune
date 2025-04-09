"""
Module for analyzing generalization performance of machine learning models.
"""

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd


def analyze_training_validation_metrics(neps_output_dir, k_folds):
    """
    Analyzes training and validation metrics across all NePS configurations and folds.
    """
    # Read the summary CSV
    summary_path = os.path.join(neps_output_dir, "summary", "full.csv")
    df = pd.read_csv(summary_path)
    
    print("\nAnalyzing training-validation generalization across all configurations and folds:")
    
    # Create output file
    output_file = Path(neps_output_dir).parent / "validation_train_generalization.txt"
    
    def log_print(message, file):
        print(message)
        file.write(message + "\n")
    
    with open(output_file, "w") as f:
        log_print(f"Analyzed {len(df)} configurations:", f)
        log_print("\n=== Final Metrics ===", f)
        
        # For each configuration
        for _, row in df.iterrows():
            config_id = row['id']
            config_dir = Path(neps_output_dir) / "configs" / f"config_{config_id}"
            
            # Skip if config directory doesn't exist
            if not config_dir.exists():
                continue
                
            log_print(f"\nConfiguration {config_id}:", f)
            # Log all configuration parameters dynamically
            for column in df.columns:
                if column.startswith('config.'):
                    param_name = column.replace('config.', '').replace('_', ' ').title()
                    log_print(f"{param_name}: {row[column]}", f)
            log_print(f"Performance: {-row['objective_to_minimize']:.2f}%", f)
            
            # Get metrics for each fold if available
            for fold in range(k_folds):
                fold_dir = config_dir / f"fold_{fold}"
                metrics_file = fold_dir / "metrics.csv"
                
                if metrics_file.exists():
                    fold_metrics = pd.read_csv(metrics_file)
                    if not fold_metrics.empty:
                        final_metrics = fold_metrics.iloc[-1]
                        log_print(f"\nFold {fold} Final Metrics:", f)
                        for metric in ['accuracy', 'loss', 'f1', 'precision', 'recall']:
                            if metric in final_metrics:
                                log_print(f"{metric.capitalize()}: {final_metrics[metric]:.2f}", f)
    
    print(f"\nGeneralization analysis saved to: {output_file}")


def analyze_validation_test_generalization(neps_output_dir, test_metrics, k_folds):
    """
    Analyzes generalization between validation and test set for the best NePS configuration.
    """
    # Read the summary CSV to get the best configuration
    summary_path = os.path.join(neps_output_dir, "summary", "full.csv")
    df = pd.read_csv(summary_path)
    
    # Get the best configuration (minimum objective_to_minimize)
    best_row = df.loc[df['objective_to_minimize'].idxmin()]
    best_config_id = best_row['id']
    
    analysis_file = Path(neps_output_dir).parent / "validation_test_generalization.txt"
    
    def log_print(message, file):
        print(message)
        file.write(message + "\n")
    
    with open(analysis_file, "w") as f:
        log_print("\n=== Validation to Test Set Generalization Analysis ===", f)
        log_print(f"\nBest Configuration (ID: {best_config_id})", f)
        
        # Get validation metrics from the best configuration's results
        config_dir = Path(neps_output_dir) / "configs" / f"config_{best_config_id}"
        val_metrics = {
            "accuracy": [],
            "loss": [],
            "f1": [],
            "precision": [],
            "recall": []
        }
        
        # Collect validation metrics from each fold
        for fold in range(k_folds):
            metrics_file = config_dir / f"fold_{fold}" / "metrics.csv"
            if metrics_file.exists():
                fold_metrics = pd.read_csv(metrics_file)
                if not fold_metrics.empty:
                    final_metrics = fold_metrics.iloc[-1]
                    for metric in val_metrics:
                        if metric in final_metrics:
                            val_metrics[metric].append(final_metrics[metric])
        
        # Calculate average validation metrics
        avg_val_metrics = {
            metric: np.mean(values) if values else 0.0
            for metric, values in val_metrics.items()
        }
        
        # Compare with test metrics
        for metric in ["accuracy", "loss", "f1", "precision", "recall"]:
            val_value = avg_val_metrics[metric]
            test_value = test_metrics.get(metric, 0.0)
            gap = val_value - test_value
            
            log_print(f"\n{metric.capitalize()}:", f)
            log_print(f"Validation: {val_value:.2f}", f)
            log_print(f"Test: {test_value:.2f}", f)
            log_print(f"Gap (Val-Test): {gap:.2f}", f)
    
    print(f"\nValidation-Test generalization analysis saved to: {analysis_file}")


def compare_validation_test_generalization(dataset, exp1, seed1, exp2, seed2):
    """
    Compare validation-test generalization between two experiments.
    """
    # Use specific seed directories
    base_path = Path("experiments") / dataset
    exp1_seed_dir = base_path / exp1 / f"seed_{seed1}"
    exp2_seed_dir = base_path / exp2 / f"seed_{seed2}"

    if not exp1_seed_dir.is_dir() or not exp2_seed_dir.is_dir():
        raise ValueError(
            f"One or both seed directories not found: {exp1_seed_dir}, {exp2_seed_dir}"
        )

    exp1_file = exp1_seed_dir / "validation_test_generalization.txt"
    exp2_file = exp2_seed_dir / "validation_test_generalization.txt"

    # Create output directory if it doesn't exist
    output_dir = base_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"validation_test_generalization_comparison_{exp1}_s{seed1}_vs_{exp2}_s{seed2}.txt"
    )

    def extract_metrics(file_path):
        metrics = {}
        with open(file_path, "r") as f:
            lines = f.readlines()
            current_metric = None
            for line in lines:
                if "Gap (Val-Test):" in line:
                    value = float(line.split(":")[1].strip())
                    if current_metric == "Accuracy":
                        metrics["accuracy_gap"] = value
                    elif current_metric == "Loss":
                        metrics["loss_gap"] = value
                    elif current_metric == "F1":
                        metrics["f1_gap"] = value
                    elif current_metric == "Precision":
                        metrics["precision_gap"] = value
                    elif current_metric == "Recall":
                        metrics["recall_gap"] = value
                elif any(metric in line for metric in ["Accuracy:", "Loss:", "F1:", "Precision:", "Recall:"]):
                    current_metric = line.split(":")[0].strip()
        
        # Ensure all metrics exist with default value of 0.0
        for metric in ["accuracy_gap", "loss_gap", "f1_gap", "precision_gap", "recall_gap"]:
            if metric not in metrics:
                metrics[metric] = 0.0
            
        return metrics

    exp1_metrics = extract_metrics(exp1_file)
    exp2_metrics = extract_metrics(exp2_file)

    def log_print(message, file):
        print(message)
        file.write(message + "\n")

    with open(output_file, "w") as f:
        log_print(f"\n=== Test-Validation Generalization Gap Comparison ===", f)
        log_print(f"Experiment 1: {exp1} (seed {seed1})", f)
        log_print(f"Experiment 2: {exp2} (seed {seed2})\n", f)

        # Compare each metric
        metrics = {
            "Accuracy": ("accuracy_gap", "%"),
            "Loss": ("loss_gap", ""),
            "F1 Score": ("f1_gap", "%"),
            "Precision": ("precision_gap", "%"),
            "Recall": ("recall_gap", "%"),
        }

        for metric_name, (metric_key, unit) in metrics.items():
            gap1 = exp1_metrics[metric_key]
            gap2 = exp2_metrics[metric_key]
            diff = (
                gap2 - gap1
            )  # Positive means exp2 has larger gap (worse generalization)

            log_print(f"{metric_name} Gap:", f)
            log_print(f"Exp1: {gap1:.2f}{unit}", f)
            log_print(f"Exp2: {gap2:.2f}{unit}", f)

            if unit == "%":
                log_print(f"Difference (Exp2 - Exp1): {diff:.2f}{unit}", f)
                if abs(diff) > 0.1:  # More than 0.1% difference
                    if diff > 0:
                        log_print(f"→ Generalization got worse by {diff:.2f}{unit}", f)
                    else:
                        log_print(
                            f"→ Generalization improved by {abs(diff):.2f}{unit}", f
                        )
            else:
                log_print(f"Difference (Exp2 - Exp1): {diff:.2f}", f)
                if abs(diff) > 0.01:  # More than 0.01 difference for loss
                    if diff > 0:
                        log_print(f"→ Generalization got worse by {diff:.2f}", f)
                    else:
                        log_print(f"→ Generalization improved by {abs(diff):.2f}", f)
            log_print("", f)

        # Overall assessment
        total_metrics = len(metrics)
        improved_metrics = sum(
            1
            for metric_key, _ in metrics.values()
            if exp2_metrics[metric_key] < exp1_metrics[metric_key]
        )

        log_print("=== Overall Assessment ===", f)
        log_print(f"Metrics improved: {improved_metrics}/{total_metrics}", f)
        if improved_metrics > total_metrics / 2:
            log_print("Overall: Generalization generally improved", f)
        elif improved_metrics < total_metrics / 2:
            log_print("Overall: Generalization generally worsened", f)
        else:
            log_print("Overall: Mixed results in generalization", f)

    print(f"\nComparison analysis saved to: {output_file}")


def compare_validation_train_generalization(dataset, exp1, seed1, exp2, seed2):
    """
    Compare validation-train generalization between two experiments.
    """
    # Use specific seed directories
    base_path = Path("experiments") / dataset
    exp1_seed_dir = base_path / exp1 / f"seed_{seed1}"
    exp2_seed_dir = base_path / exp2 / f"seed_{seed2}"

    if not exp1_seed_dir.is_dir() or not exp2_seed_dir.is_dir():
        raise ValueError(
            f"One or both seed directories not found: {exp1_seed_dir}, {exp2_seed_dir}"
        )

    exp1_file = exp1_seed_dir / "validation_train_generalization.txt"
    exp2_file = exp2_seed_dir / "validation_train_generalization.txt"

    # Create output directory if it doesn't exist
    output_dir = base_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"validation_train_generalization_comparison_{exp1}_s{seed1}_vs_{exp2}_s{seed2}.txt"
    )

    def extract_metrics(file_path):
        metrics = {}
        with open(file_path, "r") as f:
            lines = f.readlines()
            current_metric = None
            for line in lines:
                if any(metric in line for metric in ["Accuracy:", "Loss:", "F1:", "Precision:", "Recall:"]):
                    current_metric = line.split(":")[0].strip()
                elif "Gap (Train-Val):" in line or "Gap (Val-Train):" in line:
                    try:
                        # Handle both ± format and simple format
                        if "±" in line:
                            value = float(line.split("±")[0].split(":")[1].strip().rstrip("%"))
                        else:
                            value = float(line.split(":")[1].strip().rstrip("%"))
                        
                        if current_metric == "Accuracy":
                            metrics["accuracy_gap"] = value
                        elif current_metric == "Loss":
                            metrics["loss_gap"] = value
                        elif current_metric == "F1":
                            metrics["f1_gap"] = value
                        elif current_metric == "Precision":
                            metrics["precision_gap"] = value
                        elif current_metric == "Recall":
                            metrics["recall_gap"] = value
                    except (ValueError, IndexError):
                        continue
        
        # Ensure all metrics exist with default value of 0.0
        for metric in ["accuracy_gap", "loss_gap", "f1_gap", "precision_gap", "recall_gap"]:
            if metric not in metrics:
                metrics[metric] = 0.0
            
        return metrics

    exp1_metrics = extract_metrics(exp1_file)
    exp2_metrics = extract_metrics(exp2_file)

    def log_print(message, file):
        print(message)
        file.write(message + "\n")

    with open(output_file, "w") as f:
        log_print(f"\n=== Train-Validation Generalization Gap Comparison ===", f)
        log_print(f"Experiment 1: {exp1} (seed {seed1})", f)
        log_print(f"Experiment 2: {exp2} (seed {seed2})\n", f)

        # Compare each metric
        metrics = {
            "Accuracy": ("accuracy_gap", "%"),
            "Loss": ("loss_gap", ""),
            "F1 Score": ("f1_gap", "%"),
            "Precision": ("precision_gap", "%"),
            "Recall": ("recall_gap", "%"),
        }

        for metric_name, (metric_key, unit) in metrics.items():
            gap1 = exp1_metrics[metric_key]
            gap2 = exp2_metrics[metric_key]
            diff = (
                gap2 - gap1
            )  # Positive means exp2 has larger gap (worse generalization)

            log_print(f"{metric_name} Gap:", f)
            log_print(f"Exp1: {gap1:.2f}{unit}", f)
            log_print(f"Exp2: {gap2:.2f}{unit}", f)

            if unit == "%":
                log_print(f"Difference (Exp2 - Exp1): {diff:.2f}{unit}", f)
                if abs(diff) > 0.1:  # More than 0.1% difference
                    if diff > 0:
                        log_print(f"→ Generalization got worse by {diff:.2f}{unit}", f)
                    else:
                        log_print(
                            f"→ Generalization improved by {abs(diff):.2f}{unit}", f
                        )
            else:
                log_print(f"Difference (Exp2 - Exp1): {diff:.2f}", f)
                if abs(diff) > 0.01:  # More than 0.01 difference
                    if diff > 0:
                        log_print(f"→ Generalization got worse by {diff:.2f}", f)
                    else:
                        log_print(f"→ Generalization improved by {abs(diff):.2f}", f)
            log_print("", f)

        # Overall assessment
        total_metrics = len(metrics)
        improved_metrics = sum(
            1
            for metric_key, _ in metrics.values()
            if exp2_metrics[metric_key] < exp1_metrics[metric_key]
        )

        log_print("=== Overall Assessment ===", f)
        log_print(f"Metrics improved: {improved_metrics}/{total_metrics}", f)
        if improved_metrics > total_metrics / 2:
            log_print("Overall: Train-validation generalization generally improved", f)
        elif improved_metrics < total_metrics / 2:
            log_print("Overall: Train-validation generalization generally worsened", f)
        else:
            log_print("Overall: Mixed results in train-validation generalization", f)

    print(f"\nTrain-validation comparison analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze generalization performance")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--exp1", type=str, required=True, help="First experiment name")
    parser.add_argument(
        "--seed1", type=str, required=True, help="First experiment seed"
    )
    parser.add_argument(
        "--exp2", type=str, required=True, help="Second experiment name"
    )
    parser.add_argument(
        "--seed2", type=str, required=True, help="Second experiment seed"
    )

    args = parser.parse_args()

    # Run both comparisons with specific seeds
    compare_validation_test_generalization(
        args.dataset, args.exp1, args.seed1, args.exp2, args.seed2
    )
    compare_validation_train_generalization(
        args.dataset, args.exp1, args.seed1, args.exp2, args.seed2
    )


if __name__ == "__main__":
    main()
