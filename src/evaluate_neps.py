import argparse
import ast
import json
import os
import pickle
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from src.analysis.confusion_matrix import plot_confusion_matrix
# from src.test_best_config import test_run_pipeline
from src.analysis.generalization_analysis import (
    analyze_training_validation_metrics,
    analyze_validation_test_generalization)
from src.classification_2d.models_2d import get_2d_model
from src.classification_2d.preprocess_data_2d import (BrainTumorDataset,
                                                      get_max_batch_size,
                                                      load_brain_tumor_dataset)
from src.classification_3d.models_3d import get_3d_model
from src.classification_3d.preprocess_data_3d import load_3d_dataset
from src.utils.common_utils import set_seed, yaml_to_neps_pipeline_space
from src.utils.model_lifecycle_utils import evaluate_model
from src.classification_3d.preprocess_data_3d import get_kfold_dataloaders
from monai.data import Dataset
from src.classification_3d.preprocess_data_3d import EvaluationTransform


def parse_neps_results(neps_output_dir: str):
    """
    Parse the results from NePS summary CSV file.

    Args:
        neps_output_dir (str): Path to NePS output directory

    Returns:
        tuple: (dict, str) - (best hyperparameter config, config ID)
    """
    # Read the summary CSV
    summary_path = os.path.join(neps_output_dir, "summary", "full.csv")
    df = pd.read_csv(summary_path)

    # Find the best configuration (minimum objective_to_minimize)
    best_row = df.loc[df["objective_to_minimize"].idxmin()]

    # Extract config parameters dynamically
    config_params = {}
    for column in df.columns:
        if column.startswith("config."):
            param_name = column.replace("config.", "")
            value = best_row[column]
            # Convert to int if the parameter name suggests it should be an integer
            if any(int_param in param_name for int_param in ["epochs", "batch_size"]):
                value = int(value)
            config_params[param_name] = value

    # Get config ID from the 'id' column
    config_id = best_row["id"]

    print("\nBest configuration found:")
    print(f"Config ID: {config_id}")
    print("Parameters:", config_params)
    print(f"Performance: {-best_row['objective_to_minimize']:.2f}%\n")

    return config_params, config_id


def print_evaluation_results(fold_metrics, num_classes, fold_number=None):
    """
    Print detailed evaluation results including metrics and confusion matrix.

    Args:
        fold_metrics (dict): Dictionary containing evaluation metrics and confusion matrix
        num_classes (int): Number of classes in the dataset
        fold_number (int, optional): Current fold number for fold-specific output
    """
    # Print metrics
    for metric_name, metric_value in fold_metrics.items():
        if metric_name != "confusion_matrix" and metric_name != "loss":
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
        if metric_name == "loss":
            print(f"{metric_name.capitalize()}: {metric_value:.2f}")

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


def test_run_pipeline(
    _pipeline_directory,
    _previous_pipeline_directory,
    experimental_setting,
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
    set_seed(experimental_setting.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load the dataset based on dimensionality
    dimensionality = experimental_setting.data.dimensionality.lower()
    if dimensionality == "2d":
        if experimental_setting.data.dataset == "brain_tumor":
            dataset_dict = load_brain_tumor_dataset(
                data_path=experimental_setting.data.path, seed=experimental_setting.seed
            )
        else:
            raise ValueError(f"Unsupported dataset: {experimental_setting.data.dataset}.")
        num_classes = dataset_dict["num_classes"]
    elif dimensionality == "3d":  # TODO: Add 3D dataset loading
        dataset_dict = load_3d_dataset(
            experimental_setting.experiment_base_dir,
            experimental_setting.experiment_name,
            experimental_setting.data.dataset,
            data_path=experimental_setting.data.path, 
            seed=experimental_setting.seed,
            use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
            voxel_calculation=experimental_setting.data.voxel_calculation
        )
        num_classes = dataset_dict["num_classes"]
    else:
        raise ValueError(
            f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
        )

    # Convert pipeline space to NePS format (used to get the max batch size if needed)
    pipeline_space = yaml_to_neps_pipeline_space(experimental_setting.pipeline_space)

    # Create a single test loader for the complete test set
    if dimensionality == "2d":
        test_dataset = BrainTumorDataset(
            dataset_dict["test_data"], dataset_dict["test_labels"]
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=get_max_batch_size(pipeline_space),
            shuffle=False,
            num_workers=experimental_setting.data.num_workers,
        )
    elif dimensionality == "3d":  # TODO @Diane: Doublecheck this
        # For 3D datasets, we need to create the test data in the correct format
        
        # Create test data in the format expected by 3D dataloaders
        test_data_images = [{"index": idx, "image": img, "label": label} 
                           for idx, (img, label) in enumerate(zip(dataset_dict["test_data"], dataset_dict["test_labels"]))]
        
        # Get voxel size for the dataset
        voxel_size = calculate_voxel_size_from_images(
            experimental_setting.data.path, 
            experimental_setting.data.dataset, 
            calculation_method=experimental_setting.data.voxel_calculation
        )
        
        # Create test dataset with transforms (no augmentation for evaluation)
        test_dataset = Dataset(test_data_images, transform=EvaluationTransform(voxel_size, developer_mode=experimental_setting.developer_mode))
        test_loader = DataLoader(
            test_dataset,
            batch_size=get_max_batch_size(pipeline_space),
            shuffle=False,
            num_workers=experimental_setting.data.num_workers,
        )
    else:
        raise ValueError(f"Unsupported dimensionality: {dimensionality}")

    num_classes = dataset_dict["num_classes"]

    # Initialize storage for metrics across folds
    all_fold_metrics = []
    all_fold_objective_metric = []

    # Evaluate each fold's model on the complete test set
    for fold in range(k_folds):
        print(f"\n=== Evaluating Fold {fold + 1}/{k_folds} ===")

        # Initialize the model
        if dimensionality == "2d":
            model = get_2d_model(
                {
                    "type": experimental_setting.model.type,
                    "task": experimental_setting.model.task,
                    "num_classes": num_classes,
                }
            )
        elif dimensionality == "3d":
            model = get_3d_model(
                {
                    "type": experimental_setting.model.type,
                    "task": experimental_setting.model.task,
                    "num_classes": num_classes,
                }, 
                hyperparameters
            )
        else:
            raise ValueError(
                f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
            )

        # Load the trained model checkpoint for this fold
        checkpoint_path = (
            Path(neps_output_dir)
            / "configs"
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
        objective_metric = experimental_setting.metric.lower()
        all_fold_objective_metric.append(np.mean(fold_metrics[objective_metric]))

        # Print fold results using the new function
        print_evaluation_results(fold_metrics, num_classes, fold)

    # Calculate average metrics across folds dynamically
    avg_metrics = {}
    for metric_name in all_fold_metrics[0].keys():
        if metric_name == "confusion_matrix":
            avg_metrics[metric_name] = np.mean(
                [m[metric_name] for m in all_fold_metrics], axis=0
            )
        else:
            # Handle both scalar and array metrics
            values = [m[metric_name] for m in all_fold_metrics]
            if isinstance(values[0], (list, np.ndarray)):
                avg_metrics[metric_name] = np.mean([np.mean(v) for v in values]) * 100
            else:
                avg_metrics[metric_name] = np.mean(values)

    # Print average results using the same function
    print("\n=== Average Results Across All Folds ===")
    print_evaluation_results(avg_metrics, num_classes)

    return avg_metrics, num_classes


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experimental_setting.yaml",
)
def main(experimental_setting: DictConfig) -> None:
    """
    Main entry point for evaluating NePS optimization results.

    Args:
        experimental_setting (DictConfig): Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(experimental_setting.seed)

    if experimental_setting.developer_mode:
        print(f"\n\n\nDeveloper mode is enabled!\n\n\n")
        experimental_setting.data.k_folds = 2
        experimental_setting.pipeline_space = "configs/pipeline_spaces/pipeline_space_developer_mode.yaml"  # TODO @Diane: Update this
        experimental_setting.training.number_of_epochs = 2

    # Get NePS output directory from experimental setting
    neps_output_dir = os.path.join(experimental_setting.experiment_base_dir, "NePS_output")

    # Create directory for evaluation results on the test set
    test_dir = Path(experimental_setting.experiment_base_dir) / "evaluation_results"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Open text file for console output
    with open(test_dir / "evaluation_output.txt", "w") as f:
        with redirect_stdout(f):
            # Get the best hyperparameters and config ID
            best_hyperparameters, config_id = parse_neps_results(neps_output_dir)

            # Run evaluation on the test set
            avg_metrics, num_classes = test_run_pipeline(
                _pipeline_directory=str(test_dir),
                _previous_pipeline_directory=None,
                experimental_setting=experimental_setting,
                neps_output_dir=neps_output_dir,
                config_id=config_id,
                k_folds=experimental_setting.data.k_folds,
                **best_hyperparameters,
            )

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

    # Analyze generalization across all configurations
    # analyze_training_validation_metrics(neps_output_dir, experimental_setting.data.k_folds)

    # Analyze validation-test generalization
    # analyze_validation_test_generalization(neps_output_dir, avg_metrics, experimental_setting.data.k_folds)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
