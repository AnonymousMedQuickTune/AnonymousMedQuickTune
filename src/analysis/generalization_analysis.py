"""
Module for analyzing generalization performance of machine learning models.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def analyze_training_validation_metrics(neps_output_dir, k_folds):
    """
    Analyzes training and validation metrics across all NePS configurations and folds.
    """
    results_dir = Path(neps_output_dir) / "results"
    print(
        "\nAnalyzing training-validation generalization across all configurations and folds:"
    )

    all_configs_metrics_final = []
    all_configs_metrics_mean = []

    def log_print(message, file):
        print(message)
        file.write(message + "\n")

    with open(
        Path(neps_output_dir).parent / "validation_train_generalization.txt", "w"
    ) as f:
        for config_dir in results_dir.glob("config_*"):
            config_metrics_final = []
            config_metrics_mean = []

            # Analyze each fold
            for fold in range(k_folds):
                metrics_file = config_dir / f"fold_{fold}" / "logging" / "metrics.csv"
                if not metrics_file.exists():
                    continue

                df = pd.read_csv(metrics_file)

                # Get final epoch metrics
                final_epoch = df["epoch"].max()
                final_train = df[
                    (df["epoch"] == final_epoch) & (df["phase"] == "train")
                ].iloc[0]
                final_val = df[
                    (df["epoch"] == final_epoch) & (df["phase"] == "val")
                ].iloc[0]

                # Calculate mean metrics across all epochs (only for numeric columns)
                numeric_cols = ["loss", "accuracy", "precision", "recall", "f1"]
                mean_train = df[df["phase"] == "train"][numeric_cols].mean()
                mean_val = df[df["phase"] == "val"][numeric_cols].mean()

                # Calculate metrics for final epoch
                final_metrics = {
                    "config_id": config_dir.name,
                    "fold": fold,
                    "train_acc": final_train["accuracy"],
                    "val_acc": final_val["accuracy"],
                    "acc_gap": final_train["accuracy"] - final_val["accuracy"],
                    "train_loss": final_train["loss"],
                    "val_loss": final_val["loss"],
                    "loss_gap": final_val["loss"] - final_train["loss"],
                    "train_f1": final_train["f1"],
                    "val_f1": final_val["f1"],
                    "f1_gap": final_train["f1"] - final_val["f1"],
                    "train_precision": final_train["precision"],
                    "val_precision": final_val["precision"],
                    "precision_gap": final_train["precision"] - final_val["precision"],
                    "train_recall": final_train["recall"],
                    "val_recall": final_val["recall"],
                    "recall_gap": final_train["recall"] - final_val["recall"],
                }

                # Calculate metrics averaged across all epochs
                mean_metrics = {
                    "config_id": config_dir.name,
                    "fold": fold,
                    "train_acc": mean_train["accuracy"],
                    "val_acc": mean_val["accuracy"],
                    "acc_gap": mean_train["accuracy"] - mean_val["accuracy"],
                    "train_loss": mean_train["loss"],
                    "val_loss": mean_val["loss"],
                    "loss_gap": mean_val["loss"] - mean_train["loss"],
                    "train_f1": mean_train["f1"],
                    "val_f1": mean_val["f1"],
                    "f1_gap": mean_train["f1"] - mean_val["f1"],
                    "train_precision": mean_train["precision"],
                    "val_precision": mean_val["precision"],
                    "precision_gap": mean_train["precision"] - mean_val["precision"],
                    "train_recall": mean_train["recall"],
                    "val_recall": mean_val["recall"],
                    "recall_gap": mean_train["recall"] - mean_val["recall"],
                }

                config_metrics_final.append(final_metrics)
                config_metrics_mean.append(mean_metrics)

            # Average metrics across folds for this config
            if config_metrics_final:
                avg_final = {
                    key: np.mean([m[key] for m in config_metrics_final])
                    for key in config_metrics_final[0].keys()
                    if key != "config_id" and key != "fold"
                }
                avg_final["config_id"] = config_dir.name
                all_configs_metrics_final.append(avg_final)

            if config_metrics_mean:
                avg_mean = {
                    key: np.mean([m[key] for m in config_metrics_mean])
                    for key in config_metrics_mean[0].keys()
                    if key != "config_id" and key != "fold"
                }
                avg_mean["config_id"] = config_dir.name
                all_configs_metrics_mean.append(avg_mean)

        # Convert to DataFrames for analysis
        df_final = pd.DataFrame(all_configs_metrics_final)
        df_mean = pd.DataFrame(all_configs_metrics_mean)

        # Print and log final epoch statistics
        log_print(f"Analyzed {len(df_final)} configurations:", f)
        log_print("\n=== Final Epoch Metrics ===", f)
        log_print("\nAverage Metrics Across All Configs (Final Epoch):", f)
        log_print(
            f"Training Accuracy: {df_final['train_acc'].mean():.2f}% ± {df_final['train_acc'].std():.2f}",
            f,
        )
        log_print(
            f"Validation Accuracy: {df_final['val_acc'].mean():.2f}% ± {df_final['val_acc'].std():.2f}",
            f,
        )
        log_print(
            f"Accuracy Gap (Train-Val): {df_final['acc_gap'].mean():.2f}% ± {df_final['acc_gap'].std():.2f}",
            f,
        )

        log_print(
            f"\nTraining Loss: {df_final['train_loss'].mean():.2f} ± {df_final['train_loss'].std():.4f}",
            f,
        )
        log_print(
            f"Validation Loss: {df_final['val_loss'].mean():.2f} ± {df_final['val_loss'].std():.4f}",
            f,
        )
        log_print(
            f"Loss Gap (Val-Train): {df_final['loss_gap'].mean():.2f} ± {df_final['loss_gap'].std():.4f}",
            f,
        )

        log_print(
            f"\nTraining F1: {df_final['train_f1'].mean():.2f}% ± {df_final['train_f1'].std():.2f}",
            f,
        )
        log_print(
            f"Validation F1: {df_final['val_f1'].mean():.2f}% ± {df_final['val_f1'].std():.2f}",
            f,
        )
        log_print(
            f"F1 Gap (Train-Val): {df_final['f1_gap'].mean():.2f}% ± {df_final['f1_gap'].std():.2f}",
            f,
        )

        log_print(
            f"\nTraining Precision: {df_final['train_precision'].mean():.2f}% ± {df_final['train_precision'].std():.2f}",
            f,
        )
        log_print(
            f"Validation Precision: {df_final['val_precision'].mean():.2f}% ± {df_final['val_precision'].std():.2f}",
            f,
        )
        log_print(
            f"Precision Gap (Train-Val): {df_final['precision_gap'].mean():.2f}% ± {df_final['precision_gap'].std():.2f}",
            f,
        )

        log_print(
            f"\nTraining Recall: {df_final['train_recall'].mean():.2f}% ± {df_final['train_recall'].std():.2f}",
            f,
        )
        log_print(
            f"Validation Recall: {df_final['val_recall'].mean():.2f}% ± {df_final['val_recall'].std():.2f}",
            f,
        )
        log_print(
            f"Recall Gap (Train-Val): {df_final['recall_gap'].mean():.2f}% ± {df_final['recall_gap'].std():.2f}",
            f,
        )

        # Best/Worst generalizing configs based on final epoch
        best_gen_idx = df_final["acc_gap"].abs().idxmin()
        worst_gen_idx = df_final["acc_gap"].abs().idxmax()

        log_print("\n=== Best/Worst Generalizing Configurations (Final Epoch) ===", f)
        log_print("\nBest Generalizing Configuration:", f)
        log_print(f"Config ID: {df_final.loc[best_gen_idx, 'config_id']}", f)
        log_print(f"Train Acc: {df_final.loc[best_gen_idx, 'train_acc']:.2f}%", f)
        log_print(f"Val Acc: {df_final.loc[best_gen_idx, 'val_acc']:.2f}%", f)
        log_print(f"Gap: {df_final.loc[best_gen_idx, 'acc_gap']:.2f}%", f)

        log_print("\nWorst Generalizing Configuration:", f)
        log_print(f"Config ID: {df_final.loc[worst_gen_idx, 'config_id']}", f)
        log_print(f"Train Acc: {df_final.loc[worst_gen_idx, 'train_acc']:.2f}%", f)
        log_print(f"Val Acc: {df_final.loc[worst_gen_idx, 'val_acc']:.2f}%", f)
        log_print(f"Gap: {df_final.loc[worst_gen_idx, 'acc_gap']:.2f}%", f)

        # Detailed metrics for each configuration
        log_print("\n=== Detailed Metrics for All Configurations ===", f)
        for _, row in df_final.sort_values("val_acc", ascending=False).iterrows():
            config_id = row["config_id"]
            mean_row = df_mean[df_mean["config_id"] == config_id].iloc[0]

            log_print(f"\nConfig: {config_id}", f)
            log_print("Final Epoch Metrics:", f)
            log_print(f"Training Accuracy: {row['train_acc']:.2f}%", f)
            log_print(f"Validation Accuracy: {row['val_acc']:.2f}%", f)
            log_print(f"Accuracy Gap: {row['acc_gap']:.2f}%", f)
            log_print(f"Training Loss: {row['train_loss']:.4f}", f)
            log_print(f"Validation Loss: {row['val_loss']:.4f}", f)
            log_print(f"Loss Gap: {row['loss_gap']:.4f}", f)
            log_print(f"Training F1: {row['train_f1']:.2f}%", f)
            log_print(f"Validation F1: {row['val_f1']:.2f}%", f)
            log_print(f"F1 Gap: {row['f1_gap']:.2f}%", f)
            log_print(f"Training Precision: {row['train_precision']:.2f}%", f)
            log_print(f"Validation Precision: {row['val_precision']:.2f}%", f)
            log_print(f"Precision Gap: {row['precision_gap']:.2f}%", f)
            log_print(f"Training Recall: {row['train_recall']:.2f}%", f)
            log_print(f"Validation Recall: {row['val_recall']:.2f}%", f)
            log_print(f"Recall Gap: {row['recall_gap']:.2f}%", f)

            log_print("\nMean Across Epochs:", f)
            log_print(f"Training Accuracy: {mean_row['train_acc']:.2f}%", f)
            log_print(f"Validation Accuracy: {mean_row['val_acc']:.2f}%", f)
            log_print(f"Accuracy Gap: {mean_row['acc_gap']:.2f}%", f)
            log_print(f"Training Loss: {mean_row['train_loss']:.4f}", f)
            log_print(f"Validation Loss: {mean_row['val_loss']:.4f}", f)
            log_print(f"Loss Gap: {mean_row['loss_gap']:.4f}", f)
            log_print(f"Training F1: {mean_row['train_f1']:.2f}%", f)
            log_print(f"Validation F1: {mean_row['val_f1']:.2f}%", f)
            log_print(f"F1 Gap: {mean_row['f1_gap']:.2f}%", f)
            log_print(f"Training Precision: {mean_row['train_precision']:.2f}%", f)
            log_print(f"Validation Precision: {mean_row['val_precision']:.2f}%", f)
            log_print(f"Precision Gap: {mean_row['precision_gap']:.2f}%", f)
            log_print(f"Training Recall: {mean_row['train_recall']:.2f}%", f)
            log_print(f"Validation Recall: {mean_row['val_recall']:.2f}%", f)
            log_print(f"Recall Gap: {mean_row['recall_gap']:.2f}%", f)

    print(
        f"\nGeneralization analysis saved to: {Path(neps_output_dir).parent / 'validation_train_generalization.txt'}"
    )


def analyze_validation_test_generalization(neps_output_dir, test_metrics, k_folds):
    """
    Analyzes generalization between validation and test set for the best NePS configuration.
    Now considers metrics across all folds.

    Args:
        neps_output_dir (str): Path to the NePS output directory containing results
        test_metrics (dict): Dictionary containing test set metrics
    """
    results_dir = Path(neps_output_dir) / "results"
    analysis_file = Path(neps_output_dir).parent / "validation_test_generalization.txt"

    def log_print(message, file):
        print(message)
        file.write(message + "\n")

    with open(analysis_file, "w") as f:
        # Find the best configuration's metrics file
        best_config_file = (
            Path(neps_output_dir) / "best_loss_with_config_trajectory.txt"
        )
        with open(best_config_file, "r") as bcf:
            lines = bcf.readlines()
            best_config_id = None
            for line in lines:
                if line.startswith("Config ID:"):
                    best_config_id = line.replace("Config ID:", "").strip()

            if best_config_id is None:
                raise ValueError("Could not find Config ID in file")

        # Calculate average validation metrics across folds
        val_metrics = {
            metric: [] for metric in ["accuracy", "loss", "f1", "precision", "recall"]
        }

        for fold in range(k_folds):
            metrics_file = (
                results_dir
                / f"config_{best_config_id}"
                / f"fold_{fold}"
                / "logging"
                / "metrics.csv"
            )

            if not metrics_file.exists():
                continue

            df = pd.read_csv(metrics_file)
            final_epoch = df["epoch"].max()
            final_val = df[(df["epoch"] == final_epoch) & (df["phase"] == "val")].iloc[
                0
            ]

            for metric in val_metrics:
                if metric == "f1":
                    # Store F1 score directly without additional processing
                    val_metrics[metric].append(final_val[metric])
                else:
                    val_metrics[metric].append(final_val[metric])

        # Average validation metrics across folds
        avg_val_metrics = {
            metric: np.mean(values) for metric, values in val_metrics.items()
        }

        # Calculate generalization gaps
        log_print("\n=== Validation to Test Set Generalization Analysis ===", f)
        log_print(f"\nBest Configuration (ID: {best_config_id})", f)

        # Accuracy
        val_acc = avg_val_metrics["accuracy"]
        test_acc = test_metrics["accuracy"]
        acc_gap = val_acc - test_acc
        log_print(f"\nAccuracy:", f)
        log_print(f"Validation: {val_acc:.2f}%", f)
        log_print(f"Test: {test_acc:.2f}%", f)
        log_print(f"Gap (Val-Test): {acc_gap:.2f}%", f)

        # Loss
        val_loss = avg_val_metrics["loss"]
        test_loss = test_metrics["loss"]
        loss_gap = test_loss - val_loss
        log_print(f"\nLoss:", f)
        log_print(f"Validation: {val_loss:.4f}", f)
        log_print(f"Test: {test_loss:.4f}", f)
        log_print(f"Gap (Test-Val): {loss_gap:.4f}", f)

        # F1 Score
        val_f1 = avg_val_metrics["f1"]
        test_f1 = test_metrics["f1"]  # Remove np.mean() since it's already processed
        f1_gap = val_f1 - test_f1
        log_print(f"\nF1 Score:", f)
        log_print(f"Validation: {val_f1:.2f}%", f)
        log_print(f"Test: {test_f1:.2f}%", f)
        log_print(f"Gap (Val-Test): {f1_gap:.2f}%", f)

        # Precision
        val_precision = avg_val_metrics["precision"]
        test_precision = test_metrics["precision"]  # Remove np.mean()
        precision_gap = val_precision - test_precision
        log_print(f"\nPrecision:", f)
        log_print(f"Validation: {val_precision:.2f}%", f)
        log_print(f"Test: {test_precision:.2f}%", f)
        log_print(f"Gap (Val-Test): {precision_gap:.2f}%", f)

        # Recall
        val_recall = avg_val_metrics["recall"]
        test_recall = test_metrics["recall"]  # Remove np.mean()
        recall_gap = val_recall - test_recall
        log_print(f"\nRecall:", f)
        log_print(f"Validation: {val_recall:.2f}%", f)
        log_print(f"Test: {test_recall:.2f}%", f)
        log_print(f"Gap (Val-Test): {recall_gap:.2f}%", f)

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
        raise ValueError(f"One or both seed directories not found: {exp1_seed_dir}, {exp2_seed_dir}")

    exp1_file = exp1_seed_dir / "validation_test_generalization.txt"
    exp2_file = exp2_seed_dir / "validation_test_generalization.txt"

    # Create output directory if it doesn't exist
    output_dir = base_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"validation_test_generalization_comparison_{exp1}_s{seed1}_vs_{exp2}_s{seed2}.txt"

    def extract_metrics(file_path):
        metrics = {}
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Gap" in line:
                    if "Val-Test" in line:
                        value = float(line.split(":")[1].strip().rstrip("%"))
                        if "Accuracy" in lines[i - 3]:
                            metrics["accuracy_gap"] = value
                        elif "F1" in lines[i - 3]:
                            metrics["f1_gap"] = value
                        elif "Precision" in lines[i - 3]:
                            metrics["precision_gap"] = value
                        elif "Recall" in lines[i - 3]:
                            metrics["recall_gap"] = value
                    elif "Test-Val" in line and "Loss" in lines[i - 3]:
                        metrics["loss_gap"] = float(line.split(":")[1].strip())
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
        raise ValueError(f"One or both seed directories not found: {exp1_seed_dir}, {exp2_seed_dir}")

    exp1_file = exp1_seed_dir / "validation_train_generalization.txt"
    exp2_file = exp2_seed_dir / "validation_train_generalization.txt"

    # Create output directory if it doesn't exist
    output_dir = base_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"validation_train_generalization_comparison_{exp1}_s{seed1}_vs_{exp2}_s{seed2}.txt"

    def extract_metrics(file_path):
        metrics = {}
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Average Metrics Across All Configs (Final Epoch):" in line:
                    # Extract gaps from the following lines
                    for j in range(i, i + 20):  # Look at next 20 lines
                        if "Accuracy Gap (Train-Val):" in lines[j]:
                            metrics["accuracy_gap"] = float(
                                lines[j].split("±")[0].split(":")[1].strip().rstrip("%")
                            )
                        elif "Loss Gap (Val-Train):" in lines[j]:
                            metrics["loss_gap"] = float(
                                lines[j].split("±")[0].split(":")[1].strip()
                            )
                        elif "F1 Gap (Train-Val):" in lines[j]:
                            metrics["f1_gap"] = float(
                                lines[j].split("±")[0].split(":")[1].strip().rstrip("%")
                            )
                        elif "Precision Gap (Train-Val):" in lines[j]:
                            metrics["precision_gap"] = float(
                                lines[j].split("±")[0].split(":")[1].strip().rstrip("%")
                            )
                        elif "Recall Gap (Train-Val):" in lines[j]:
                            metrics["recall_gap"] = float(
                                lines[j].split("±")[0].split(":")[1].strip().rstrip("%")
                            )
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
    parser.add_argument("--seed1", type=str, required=True, help="First experiment seed")
    parser.add_argument("--exp2", type=str, required=True, help="Second experiment name")
    parser.add_argument("--seed2", type=str, required=True, help="Second experiment seed")

    args = parser.parse_args()

    # Run both comparisons with specific seeds
    compare_validation_test_generalization(args.dataset, args.exp1, args.seed1, args.exp2, args.seed2)
    compare_validation_train_generalization(args.dataset, args.exp1, args.seed1, args.exp2, args.seed2)


if __name__ == "__main__":
    main()
