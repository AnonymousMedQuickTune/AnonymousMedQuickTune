import argparse
import os
from pathlib import Path
import json

import pandas as pd
from omegaconf import OmegaConf

# from src.test_best_config import test_run_pipeline
from src.analysis.generalization_analysis import (
    analyze_training_validation_metrics,
    analyze_validation_test_generalization
)
import traceback


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
from src.classification_2d.models_2d import get_2d_model
from src.classification_3d.models_3d import get_3d_model
from src.classification_2d.preprocess_data_2d import (BrainTumorDataset,
                                                      get_max_batch_size,
                                                      load_brain_tumor_dataset)
from src.utils.common_utils import set_seed, yaml_to_neps_pipeline_space
from src.utils.model_lifecycle_utils import evaluate_model

import hydra
from omegaconf import DictConfig

from src.classification_3d.preprocess_data_3d import load_3d_dataset

from src.analysis.confusion_matrix import plot_confusion_matrix


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
    best_row = df.loc[df['objective_to_minimize'].idxmin()]
    
    # Extract config parameters dynamically
    config_params = {}
    for column in df.columns:
        if column.startswith('config.'):
            param_name = column.replace('config.', '')
            value = best_row[column]
            # Convert to int if the parameter name suggests it should be an integer
            if any(int_param in param_name for int_param in ['epochs', 'batch_size']):
                value = int(value)
            config_params[param_name] = value
    
    # Get config ID from the 'id' column
    config_id = best_row['id']
    
    print("\nBest configuration found:")
    print(f"Config ID: {config_id}")
    print("Parameters:", config_params)
    print(f"Performance: {-best_row['objective_to_minimize']:.2f}%\n")
    
    return config_params, config_id

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

    # Load the dataset based on dimensionality
    dimensionality = config.data.dimensionality.lower()
    if dimensionality == "2d":
        if config.data.dataset == "brain_tumor":
            dataset_dict = load_brain_tumor_dataset(
                data_path=config.data.path, seed=config.seed
            )
        else:
            raise ValueError(f"Unsupported dataset: {config.data.dataset}.")
        num_classes = dataset_dict["num_classes"]
    elif dimensionality == "3d":  # TODO: Add 3D dataset loading
        dataset_dict = load_3d_dataset(
            config.data.dataset, data_path=config.data.path, seed=config.seed
        )
        num_classes = dataset_dict["num_classes"]
    else:
        raise ValueError(
            f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
        )

    # Convert pipeline space to NePS format (used to get the max batch size if needed)
    pipeline_space = yaml_to_neps_pipeline_space(config.pipeline_space)

    # Create a single test loader for the complete test set
    test_dataset = BrainTumorDataset(dataset_dict["test_data"], dataset_dict["test_labels"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=get_max_batch_size(pipeline_space),
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    num_classes = dataset_dict["num_classes"]

    # Initialize storage for metrics across folds
    all_fold_metrics = []
    all_fold_f1_scores = []

    # Evaluate each fold's model on the complete test set
    for fold in range(k_folds):
        print(f"\n=== Evaluating Fold {fold + 1}/{k_folds} ===")

        # Initialize the model
        if dimensionality == "2d":
            model = get_2d_model(
                {
                    "type": config.model.type,
                    "task": config.model.task,
                    "num_classes": num_classes,
                }
            )
        elif dimensionality == "3d":
            model = get_3d_model(
                {
                    "type": config.model.type,
                    "task": config.model.task,
                    "num_classes": num_classes,
                }
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
    # analyze_validation_test_generalization(neps_output_dir, avg_metrics, k_folds)

    return avg_metrics

@hydra.main(
    version_base=None, 
    config_path="../configs", 
    config_name="main_experiment_config.yaml"
)
def main(config: DictConfig) -> None:
    """
    Main entry point for evaluating NePS optimization results.
    
    Args:
        config (DictConfig): Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(config.seed)

    # Get NePS output directory from config
    neps_output_dir = os.path.join(config.experiment_base_dir, "NePS_output")
    
    # Get the best hyperparameters and config ID
    best_hyperparameters, config_id = parse_neps_results(neps_output_dir)

    # Create directory for evaluation results on the test set
    test_dir = Path(config.experiment_base_dir) / "evaluation_results"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation on the test set with configuration that achieved the best performance on the validation set
    avg_metrics = test_run_pipeline(
        _pipeline_directory=str(test_dir),
        _previous_pipeline_directory=None,
        config=config,
        neps_output_dir=neps_output_dir,
        config_id=config_id,
        k_folds=config.data.k_folds,
        **best_hyperparameters,
    )

    # Convert NumPy arrays to lists for JSON serialization
    json_compatible_metrics = {}
    for key, value in avg_metrics.items():
        if key == 'confusion_matrix':
            json_compatible_metrics[key] = value.tolist()
        else:
            json_compatible_metrics[key] = float(value)
            
    # Save the evaluation results to a JSON file
    results_path = test_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(json_compatible_metrics, f, indent=4)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        conf_matrix=avg_metrics['confusion_matrix'],
        metrics=avg_metrics,
        save_path=test_dir / "confusion_matrix.pdf"
    )

    # Analyze generalization across all configurations
    # analyze_training_validation_metrics(neps_output_dir, config.data.k_folds)

    # Analyze validation-test generalization
    # analyze_validation_test_generalization(neps_output_dir, avg_metrics, config.data.k_folds)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
