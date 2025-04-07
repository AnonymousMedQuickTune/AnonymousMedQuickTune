import argparse
import ast
import os
import pickle
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from src.analysis.generalization_analysis import (
    analyze_training_validation_metrics,
    analyze_validation_test_generalization)
from src.classification_2d.preprocess_data_2d import load_brain_tumor_dataset, BrainTumorDataset, get_max_batch_size
from src.classification_2d.models_2d import get_model
from src.utils.common_utils import set_seed
from src.utils.model_lifecycle_utils import evaluate_model

from src.utils.common_utils import yaml_to_neps_pipeline_space


def parse_best_config(config_file_path):
    """
    Parse the best configuration and config ID from the NePS output file.
    Always returns the last configuration found in the file.

    Args:
        config_file_path (str): Path to the best_loss_with_config_trajectory.txt file

    Returns:
        tuple: (dict, str) - (best hyperparameter config, config ID)

    Raises:
        ValueError: If no Config ID or Config was found in the file
    """
    with open(config_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    last_config = None
    last_config_id = None

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Config ID:"):
            last_config_id = line.replace("Config ID:", "").strip()
            # Get the config from the next line
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("Config:"):
                last_config = lines[i + 1].replace("Config:", "").strip()

    if last_config is None or last_config_id is None:
        raise ValueError("Could not find Config ID or Config in file")

    print("\n\nEvaluating best NePS config on the test set:\n", last_config, "\n\n")
    return ast.literal_eval(last_config), last_config_id


def test_run_pipeline(
    _pipeline_directory,
    _previous_pipeline_directory,
    config,
    neps_output_dir,
    config_id,
    k_folds,
    **hyperparameters,
):
    """
    Runs a test evaluation with the best hyperparameters on the test set for each fold.
    Each fold's model is evaluated on the complete test set.
    """
    # Set seed for pipeline reproducibility
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load the dataset first
    dataset = load_brain_tumor_dataset(data_path=config.data.path)
    pipeline_space = yaml_to_neps_pipeline_space(config.pipeline_space)

    # Create a single test loader for the complete test set
    test_dataset = BrainTumorDataset(dataset["test_data"], dataset["test_labels"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=get_max_batch_size(pipeline_space),
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    num_classes = dataset["num_classes"]

    # Initialize storage for metrics across folds
    all_fold_metrics = []
    all_fold_f1_scores = []

    # Evaluate each fold's model on the complete test set
    for fold in range(k_folds):
        print(f"\n=== Evaluating Fold {fold + 1}/{k_folds} ===")

        # Initialize the model
        model = get_model(
            {
                "type": config.model.type,
                "task": config.model.task,
                "num_classes": num_classes,
            }
        )

        # Load the trained model checkpoint for this fold
        checkpoint_path = (
            Path(neps_output_dir)
            / "results"
            / f"config_{config_id}"
            / f"fold_{fold}"
            / "model_latest_checkpoint.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss(
            label_smoothing=hyperparameters.get("label_smoothing", 0.0)
        )

        # Test evaluation for this fold's model on complete test set
        fold_metrics = evaluate_model(model, test_loader, criterion, device)
        all_fold_metrics.append(fold_metrics)
        all_fold_f1_scores.append(np.mean(fold_metrics["f1"]))

        # Print fold results
        print(f"\nFold {fold + 1} Results:")
        print(f"Loss: {fold_metrics['loss']:.4f}")
        print(f"Accuracy: {fold_metrics['accuracy']:.2f}%")
        print(f"Precision: {np.mean(fold_metrics['precision'])*100:.2f}%")
        print(f"Recall: {np.mean(fold_metrics['recall'])*100:.2f}%")
        print(f"F1-Score: {np.mean(fold_metrics['f1'])*100:.2f}%")

        # Detailed output of the confusion matrix
        conf_matrix = np.array(fold_metrics["confusion_matrix"])
        total_samples = np.sum(conf_matrix)
        print(f"\nConfusion Matrix (Total samples: {total_samples}):")
        print("Predicted →      Class 0    Class 1")
        print("Actual ↓")
        class0_total = conf_matrix[0, 0] + conf_matrix[0, 1]
        class1_total = conf_matrix[1, 0] + conf_matrix[1, 1]
        print(
            f"Class 0      {conf_matrix[0,0]:>10d} {conf_matrix[0,1]:>10d}    | "
            f"{class0_total:>3d} total"
        )
        print(
            f"Class 1      {conf_matrix[1,0]:>10d} {conf_matrix[1,1]:>10d}    | "
            f"{class1_total:>3d} total"
        )
        print("            ----------------------")
        print(
            f"Total        {conf_matrix[:,0].sum():>10d} {conf_matrix[:,1].sum():>10d}"
        )

        print("\nDetailed Interpretation:")
        print(
            f"True Negatives (TN)  : {conf_matrix[0,0]} (Correctly predicted Class 0)"
        )
        print(
            f"False Positives (FP) : {conf_matrix[0,1]} (Class 0 wrongly predicted as Class 1)"
        )
        print(
            f"False Negatives (FN) : {conf_matrix[1,0]} (Class 1 wrongly predicted as Class 0)"
        )
        print(
            f"True Positives (TP)  : {conf_matrix[1,1]} (Correctly predicted Class 1)"
        )

    # Calculate average metrics across folds
    avg_metrics = {
        "loss": np.mean([m["loss"] for m in all_fold_metrics]),
        "accuracy": np.mean([m["accuracy"] for m in all_fold_metrics]),
        "precision": np.mean([np.mean(m["precision"]) for m in all_fold_metrics]) * 100,
        "recall": np.mean([np.mean(m["recall"]) for m in all_fold_metrics]) * 100,
        "f1": np.mean(all_fold_f1_scores) * 100,
        "confusion_matrix": np.mean(
            [m["confusion_matrix"] for m in all_fold_metrics], axis=0
        ),
    }

    # Print average results
    print("\n=== Average Results Across All Folds ===")
    print(f"Loss: {avg_metrics['loss']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.2f}%")
    print(f"Precision: {avg_metrics['precision']:.2f}%")
    print(f"Recall: {avg_metrics['recall']:.2f}%")
    print(f"F1-Score: {avg_metrics['f1']:.2f}%")

    # Analyze validation-test generalization
    analyze_validation_test_generalization(neps_output_dir, avg_metrics, k_folds)

    return avg_metrics


def main():
    """
    Main function to run the test evaluation pipeline.

    Loads the best configuration from NePS output, initializes the model with the best
    hyperparameters, and evaluates it on the test set. Saves the test performance
    to a file in the same directory as the model checkpoint.
    """
    parser = argparse.ArgumentParser(
        description="Train model with optimal hyperparameters"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to best_loss_with_config_trajectory.txt",
    )
    parser.add_argument(
        "--hydra_config",
        type=str,
        required=True,
        help="Path to hydra config file (e.g., main_experiment_config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to override config",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        required=True,
        help="Number of folds for cross-validation",
    )
    args = parser.parse_args()

    # Load the hydra config
    config = OmegaConf.load(args.hydra_config)

    # Override the dataset and data path in config
    config.data.dataset = args.dataset
    config.data.path = args.data_dir

    # Get the best hyperparameters and config ID
    best_hyperparameters, config_id = parse_best_config(args.config_path)

    # Get NePS output directory from config_path
    neps_output_dir = Path(args.config_path).parent

    # Analyze generalization across all configurations
    analyze_training_validation_metrics(neps_output_dir, args.k_folds)

    # Create test directory
    test_dir = Path(config.experiment_base_dir) / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save the performance file one level above NePS_output
    performance_file = neps_output_dir.parent / "test_performance.txt"

    # Capture all output in the file while also printing to console
    with performance_file.open("w") as f:
        with redirect_stdout(f):
            test_run_pipeline(
                _pipeline_directory=str(test_dir),
                _previous_pipeline_directory=None,
                config=config,
                neps_output_dir=neps_output_dir,
                config_id=config_id,
                k_folds=args.k_folds,
                **best_hyperparameters,
            )

    print(f"\nResults saved to: {performance_file}")


if __name__ == "__main__":
    main()
