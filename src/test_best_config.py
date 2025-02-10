"""
Test script to train model with optimal hyperparameters found by NePS.
"""

import argparse
import ast
from contextlib import redirect_stdout
from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn

from src.analysis.generalization_analysis import (
    analyze_training_validation_metrics,
    analyze_validation_test_generalization)
from src.data import get_data_loaders
from src.util_functions import evaluate_model, get_model, set_seed


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
    _pipeline_directory,  # Unused but required by pipeline interface
    _previous_pipeline_directory,  # Unused but required by pipeline interface
    config,
    neps_output_dir,
    config_id,
    **hyperparameters,
):
    """
    Runs a test evaluation with the best hyperparameters on the test set.

    Args:
        _pipeline_directory (str): Required by pipeline interface but not used
        _previous_pipeline_directory (str): Required by pipeline interface but not used
        config (OmegaConf): Configuration object with model and data settings
        neps_output_dir (str): Directory containing the NePS output files
        config_id (str): ID of the configuration to test
        **hyperparameters: Best hyperparameters found by NePS

    Returns:
        dict: Dictionary containing the negative test accuracy as 'loss'
    """
    # Set seed for pipeline reproducibility
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load normalization stats from cache
    cache_dir = os.path.join(config.data.path, "cache")
    cache_file = os.path.join(cache_dir, f"{config.data.dataset}_normalization_stats.pkl")
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Normalization stats cache not found at {cache_file}. "
            "Please run preprocess_dataset.py first."
        )
    
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)
        normalization_stats = cached_data["normalization_stats"]

    # Load test dataset and create data loader with cached normalization stats
    test_loader, num_classes = get_data_loaders(
        config.data.dataset,
        config.data.num_workers,
        hyperparameters["batch_size"],
        split="test",
        data_path=config.data.path,
        normalization_stats=normalization_stats  # Pass the cached stats
    )
    print(f"Test dataset '{config.data.dataset}' loaded with {num_classes} classes")
    print(f"Test batches: {len(test_loader)}\n")

    # Initialize the model
    model = get_model(
        {
            "type": config.model.type,
            "task": config.model.task,
            "num_classes": num_classes,
        }
    )

    # Load the trained model checkpoint
    checkpoint_path = (
        Path(neps_output_dir)
        / "results"
        / f"config_{config_id}"
        / "model_latest_checkpoint.pth"
    )
    print(f"\n\nCheckpoint path: {checkpoint_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded: {config.model.type}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=hyperparameters["label_smoothing"])

    # Test evaluation
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    # Convert metrics to percentages right after evaluation
    test_metrics["precision"] = [p * 100 for p in test_metrics["precision"]]
    test_metrics["recall"] = [r * 100 for r in test_metrics["recall"]]
    test_metrics["f1"] = [f * 100 for f in test_metrics["f1"]]

    # Analyze validation-test generalization
    analyze_validation_test_generalization(neps_output_dir, test_metrics)

    print("\nTest Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Precision: {np.mean(test_metrics['precision']):.2f}%")
    print(f"Recall: {np.mean(test_metrics['recall']):.2f}%")
    print(f"F1-Score: {np.mean(test_metrics['f1']):.2f}%")

    print("\nPer-class metrics:")
    for i, (p, r, f1) in enumerate(
        zip(test_metrics["precision"], test_metrics["recall"], test_metrics["f1"])
    ):
        print(f"Class {i}:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall: {r:.4f}")
        print(f"  F1-Score: {f1:.4f}")

    # Detailed output of the confusion matrix
    conf_matrix = np.array(test_metrics["confusion_matrix"])
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
    print(f"Total        {conf_matrix[:,0].sum():>10d} {conf_matrix[:,1].sum():>10d}")

    print("\nDetailed Interpretation:")
    print(f"True Negatives (TN)  : {conf_matrix[0,0]} (Correctly predicted Class 0)")
    print(
        f"False Positives (FP) : {conf_matrix[0,1]} (Class 0 wrongly predicted as Class 1)"
    )
    print(
        f"False Negatives (FN) : {conf_matrix[1,0]} (Class 1 wrongly predicted as Class 0)"
    )
    print(f"True Positives (TP)  : {conf_matrix[1,1]} (Correctly predicted Class 1)")

    # return {"loss": -test_metrics["accuracy"]}  # NePS minimizes negative accuracy


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
    args = parser.parse_args()

    # Load the hydra config
    config = OmegaConf.load(args.hydra_config)

    # Override the dataset in config with the one provided via command line
    config.data.dataset = args.dataset

    # Get the best hyperparameters and config ID
    best_hyperparameters, config_id = parse_best_config(args.config_path)

    # Get NePS output directory from config_path
    neps_output_dir = Path(args.config_path).parent

    # Analyze generalization across all configurations
    analyze_training_validation_metrics(neps_output_dir)

    # Create test directory
    test_dir = Path(config.experiment_base_dir) / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Änderung: Speichere die Performance-Datei eine Ebene höher als NePS_output
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
                **best_hyperparameters,
            )

    print(f"\nResults saved to: {performance_file}")


if __name__ == "__main__":
    main()
