import os

import numpy as np
import torch

from src.classification_3d.models_3d import \
    get_3d_model  # TODO: change to 3d models
from src.utils.common_utils import set_seed
from src.utils.logging_utils import (initialize_logging_files, log_gradients,
                                     log_initial_state, log_learning_rate,
                                     log_metrics, log_resources, log_timing)
from src.utils.model_lifecycle_utils import get_optimizer


def run_3d_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    config,
    dataset_dict=None,
    num_classes=None,
    **hyperparameters,
):
    """
    Main training pipeline for 3D data model optimization using NePS with K-Fold Cross Validation.

    NOTE: The argument order and parameter names must strictly follow NePS conventions
    to ensure proper optimization and checkpointing functionality.

    Args:
        pipeline_directory (str): Directory where current pipeline results will be saved
        previous_pipeline_directory (str): Directory containing previous pipeline runs
        config (DictConfig): Hydra configuration object
        dataset_dict (dict, optional): Combined train+val data and labels dictionary if preloaded
        num_classes (int, optional): Number of classes in the dataset if preloaded
        **hyperparameters: Configuration dictionary containing hyperparameters

    Returns:
        dict: Dictionary containing:
            - objective_to_minimize (float): Negative mean of selected metric (K-fold avg) for NePS optimization
            - cost (float): Cost of the pipeline (optional)
            - extra (dict): Dictionary containing:
                - selected_metric (float): Mean of selected metric (K-fold avg)
                - all_folds_final_metrics (dict): Dictionary containing the mean value for each
                  metric across all folds
    """
    # Set seed for pipeline reproducibility
    set_seed(
        config.seed
    )  # For more details on the config: pls see configs/main_experiment.yaml

    # Set device (GPU/CPU) for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Add 3D trainings pipeline -------------------------------------------------------------
    # Reference implementation available in: src/classification_2d/objective_function_2d.py

    # Placeholder:
    num_classes = 2
    train_time = 20
    eval_time = 10
    epoch_time = 30
    model = get_3d_model(
        {
            "type": config.model.type,
            "task": config.model.task,
            "num_classes": num_classes,
        }
    ).to(device)
    epoch = 10

    # Example: How to access hyperparameters from a configuration
    # For more details on the search spaces: pls see configs/pipeline_configs/
    optimizer = get_optimizer(
        model=model,
        optimizer_type=hyperparameters.get("optimizer_type", "adam"),
        # Get learning_rate from hyperparameters if 'learning_rate' exists in the search space,
        # otherwise use default value of 0.001
        learning_rate=hyperparameters.get("learning_rate", 1e-3),
        weight_decay=hyperparameters.get("weight_decay", 0.0),
    )

    # Example: Logging 5-fold cross validation for NePS
    k_folds = 5
    for fold in range(k_folds):
        print(f"\nTraining Fold {fold + 1}/{k_folds}\n... training...")

        # Create fold-specific directory
        fold_directory = os.path.join(pipeline_directory, f"fold_{fold}")
        os.makedirs(fold_directory, exist_ok=True)

        # Initialize logging files for this fold
        logging_dir = os.path.join(fold_directory, "logging")
        log_files = initialize_logging_files(logging_dir)

        # ... Training ...

        # Log all metrics and information at the end of the epoch
        log_timing(log_files["timing"], epoch, train_time, eval_time, epoch_time)
        log_learning_rate(log_files["lr"], epoch, optimizer)
        log_resources(log_files["resource"], epoch)
        # ...
    print("\nTraining completed!")

    # Placeholder for metric values: The metrics from each of the 5 folds in the last epoch
    all_folds_final_metrics = {
        "accuracy": [90, 87, 85, 89, 88],
        "precision": [90, 87, 86, 89, 88],
        "recall": [90, 87, 82, 89, 88],
        "f1": [90, 87, 83, 89, 88],
    }
    # --------------------------------------------------------------------------------------------

    # For NePS:
    # NePS requires a single objective (loss) to minimize. We use the negative of one selected
    # metric (e.g., f1-score) as the loss. Additional metrics are logged in 'info_dict'.

    # Get the average of the selected metric across all K-folds
    selected_metric = np.mean(all_folds_final_metrics[config.metric])
    print(f"\nSelected metric ({config.metric}): {selected_metric:.2f}%\n")

    # Convert to NePS loss (negative because NePS minimizes)
    neps_loss = -selected_metric

    # Add cost calculation (optional, currently unused feature in this project)
    # If we want to stop a NePS run after a certain total max_cost_toal is reached, we can define
    # the cost of one config evaluation, e.g. the time it takes to run a k-fold cv on one config.
    cost = epoch_time

    return {
        "objective_to_minimize": neps_loss,  # Required by NePS
        "cost": cost,
        "extra": {  # Additional information
            "selected_metric": selected_metric,
            "all_folds_final_metrics": {
                metric: np.mean(values)
                for metric, values in all_folds_final_metrics.items()
            },
        },
    }
