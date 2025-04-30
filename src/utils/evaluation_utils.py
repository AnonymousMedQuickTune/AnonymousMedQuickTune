from pathlib import Path
import numpy as np

import json
from src.analysis.confusion_matrix import plot_confusion_matrix

def save_evaluation_results(avg_metrics: dict, test_dir: Path, num_classes: int):
    """
    Save evaluation results and visualizations.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        output_dir (Path): Directory to save results
        num_classes (int): Number of classes in the dataset
    """    
    # Convert NumPy arrays to lists for JSON serialization
    json_compatible_metrics = {}
    for key, value in avg_metrics.items():
        if key == "confusion_matrix":
            json_compatible_metrics[key] = value.tolist()
        else:
            json_compatible_metrics[key] = float(value)

    # Save the evaluation results to a JSON file
    results_path = test_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(json_compatible_metrics, f, indent=4)

    # Plot and save confusion matrix
    plot_confusion_matrix(
        conf_matrix=avg_metrics["confusion_matrix"],
        metrics=avg_metrics,
        class_names=[
            f"Class {i}" for i in range(num_classes)
        ],  # Add class names dynamically
        save_path=test_dir / "confusion_matrix.pdf",
    )

def print_evaluation_results(fold_metrics, num_classes, fold_number=None):
    """
    Print detailed evaluation results including metrics and confusion matrix.

    Args:
        fold_metrics (dict): Dictionary containing evaluation metrics and confusion matrix
        num_classes (int): Number of classes in the dataset
        fold_number (int, optional): Current fold number for fold-specific output
    """
    fold_str = f" (Fold {fold_number + 1})" if fold_number is not None else ""
    print(f"\nEvaluation Results{fold_str}:")

    # Print metrics
    for metric_name, metric_value in fold_metrics.items():
        if metric_name not in ["confusion_matrix", "loss", "predicted_performance", "predicted_cost"]:
            if fold_number is not None:
                print(f"{metric_name.capitalize()}: {np.mean(metric_value)*100:.2f}%")
            else:
                # For the average metrics, we don't need to multiply by 100
                if metric_name == "accuracy":  # TODO: fix hardcoding for accuracy
                    print(
                        f"{metric_name.capitalize()}: {np.mean(metric_value)*100:.2f}%"
                    )
                else:
                    print(f"{metric_name.capitalize()}: {np.mean(metric_value):.2f}%")
        elif metric_name == "loss":
            print(f"{metric_name.capitalize()}: {metric_value:.2f}")
        elif metric_name == "predicted_performance":
            print(f"Predicted Performance: {metric_value:.2f}%")
        elif metric_name == "predicted_cost":
            print(f"Predicted Cost: {metric_value:.2f}")

    # Print confusion matrix
    conf_matrix = np.array(fold_metrics["confusion_matrix"])
    total_samples = np.sum(conf_matrix)
    print(f"\nConfusion Matrix (Total samples: {total_samples:.1f}):")

    # Header
    header = "Predicted →"
    for i in range(num_classes):
        header += f"    Class {i:2d}"
    print(header)
    print("Actual ↓")

    # Matrix rows with class totals
    for i in range(num_classes):
        row = f"Class {i:2d}   "
        for j in range(num_classes):
            row += f" {conf_matrix[i,j]:8.1f}"
        class_total = conf_matrix[i, :].sum()
        row += f"    | {class_total:5.1f} total"
        print(row)

    print("          " + "-" * (10 * num_classes))

    # Column totals
    total_row = "Total      "
    for j in range(num_classes):
        total_row += f" {conf_matrix[:,j].sum():8.1f}"
    print(total_row)

    # Detailed interpretation
    print("\nDetailed Interpretation:")
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                print(
                    f"True Class {i} (T{i})     : {conf_matrix[i,i]:.1f} "
                    f"(Correctly predicted Class {i})"
                )
            else:
                print(
                    f"Class {i} as Class {j}    : {conf_matrix[i,j]:.1f} "
                    f"(Class {i} wrongly predicted as Class {j})"
                )