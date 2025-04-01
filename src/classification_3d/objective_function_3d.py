from src.utils.common_utils import set_seed
import numpy as np
from omegaconf import DictConfig

def run_3d_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    config,
    dataset_dict,
    num_classes,
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
            - loss (float): Negative mean of selected metric (K-fold avg) for NePS optimization
            - info_dict (dict): Dictionary containing:
                - selected_metric (float): Mean of selected metric (K-fold avg)
                - all_folds_final_metrics (dict): Dictionary containing the mean value for each metric across all folds
            - cost (float): Cost of the pipeline (optional)
    """
    # Set seed for pipeline reproducibility
    set_seed(config.seed)

    # TODO: Add 3D trainings pipeline

    # TODO: This is a placeholder. Replace it with actual metric values
    all_folds_final_metrics = {
        "accuracy": [90, 87, 85, 89, 88],
        "precision": [90, 87, 86, 89, 88],
        "recall": [90, 87, 82, 89, 88],
        "f1": [90, 87, 83, 89, 88]
    }

    # Get the specified metric from final metrics for NePS
    selected_metric = np.mean(all_folds_final_metrics[config.metric])
    print(f"\nSelected metric ({config.metric}): {selected_metric:.2f}%\n")

    # Convert to NePS loss (negative because NePS minimizes)
    neps_loss = -selected_metric

    # TODO: Add cost calculation (optional)
    cost = 0

    return {
        "loss": neps_loss,
        "info_dict": {
            "selected_metric": np.mean(all_folds_final_metrics[config.metric]),
            "all_folds_final_metrics": {
                metric: np.mean(values) 
                for metric, values in all_folds_final_metrics.items()
            },
        },
        "cost": cost,
    }