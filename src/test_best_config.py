"""
Test script to train model with optimal hyperparameters found by NePS.
"""

import argparse
import ast
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

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

    # Load test dataset and create data loader
    test_loader, _, num_classes = get_data_loaders(
        config.data.dataset,
        config.data.num_workers,
        hyperparameters["batch_size"],
        split="test",
        data_path=config.data.path,
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

    # Use evaluate_model from util_functions instead of local implementation
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

    print(f"\nTest metrics - Loss: {test_loss:.4f}, Acc: {test_accuracy:.2f}%")

    return {"loss": test_accuracy}


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
        help="Path to hydra config file (e.g., desmoid_config.yaml)",
    )
    args = parser.parse_args()

    # Load the hydra config
    config = OmegaConf.load(args.hydra_config)

    # Get the best hyperparameters and config ID
    best_hyperparameters, config_id = parse_best_config(args.config_path)

    # Create test directory
    test_dir = Path(config.experiment_base_dir) / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Get NePS output directory from config_path
    neps_output_dir = Path(args.config_path).parent

    # Run testing with best hyperparameters on test set
    result = test_run_pipeline(
        _pipeline_directory=str(test_dir),
        _previous_pipeline_directory=None,
        config=config,
        neps_output_dir=neps_output_dir,
        config_id=config_id,
        **best_hyperparameters,
    )

    # Calculate final accuracy
    final_accuracy = result["loss"]
    print(f"\nTest run completed with final accuracy: {final_accuracy:.2f}%")

    # Save test performance to file
    results_dir = Path(neps_output_dir) / "results" / f"config_{config_id}"
    performance_file = results_dir / "test_set_performance.txt"
    performance_file.write_text(f"{final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
