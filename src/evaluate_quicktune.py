import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from src.analysis.confusion_matrix import plot_confusion_matrix
from src.classification_2d.models_2d import get_2d_model
from src.classification_2d.preprocess_data_2d import (BrainTumorDataset,
                                                    get_max_batch_size,
                                                    load_brain_tumor_dataset)
from src.classification_3d.preprocess_data_3d import load_3d_dataset
from src.utils.common_utils import set_seed, yaml_to_neps_pipeline_space
from src.utils.model_lifecycle_utils import evaluate_model
from src.utils.evaluation_utils import save_evaluation_results, print_evaluation_results

from qtt import CostPredictor, PerfPredictor, Predictor

from src.analysis.generalization_analysis import (
    analyze_training_validation_metrics,
    analyze_validation_test_generalization)


def parse_quicktune_results(quicktune_output_dir: str):
    """
    Parse the results from Quicktune output directory.

    Args:
        quicktune_output_dir (str): Path to Quicktune output directory

    Returns:
        tuple: (dict, str, float, float) - (best hyperparameter config, config ID, score, cost)
    """
    # Read the incumbent JSON file
    incumbent_path = os.path.join(quicktune_output_dir, "tuner", "incumbent.json")
    with open(incumbent_path, 'r') as f:
        incumbent_data = json.load(f)

    # Read the history CSV file to get the config ID
    history_path = os.path.join(quicktune_output_dir, "tuner", "history.csv")
    history_df = pd.read_csv(history_path)
    
    # Extract config parameters, score, and cost
    config_params = incumbent_data["config"]
    score = incumbent_data["score"]
    cost = incumbent_data["cost"]
    
    # Find the matching configuration in history and get its config ID
    for _, row in history_df.iterrows():
        if row['score'] == score and row['cost'] == cost:
            config_id = str(row['config-id'])
            break
    else:
        config_id = "incumbent"  # Fallback if no match found

    print("\nBest configuration found:")
    print(f"Config ID: {config_id}")
    print("Parameters:", config_params)
    print(f"Performance: {score:.2f}%")
    print(f"Cost: {cost:.2f}\n")

    return config_params, config_id, score, cost


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main_experiment_config.yaml",
)


def test_run_pipeline(
    _pipeline_directory,
    _previous_pipeline_directory,
    config,
    quicktune_output_dir,
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
    test_dataset = BrainTumorDataset(
        dataset_dict["test_data"], dataset_dict["test_labels"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=get_max_batch_size(pipeline_space),
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    num_classes = dataset_dict["num_classes"]



    # ----------
    num_classes = dataset_dict["num_classes"]
    all_fold_metrics = []
    predicted_performances = []
    predicted_costs = []

    # Initialize predictors with proper paths
    predictor_path = Path(quicktune_output_dir) / "predictors"
    
    try:
        # Load predictors and verify they are fitted
        perf_predictor = PerfPredictor.load(str(predictor_path / "perf"), verbose=True)
        cost_predictor = CostPredictor.load(str(predictor_path / "cost"), verbose=True)
        
        if not perf_predictor.is_fit or not cost_predictor.is_fit:
            raise RuntimeError("Predictors must be fitted before evaluation")

        # Reset paths to evaluation directory and save
        eval_predictor_path = Path(_pipeline_directory) / "predictors"
        eval_predictor_path.mkdir(parents=True, exist_ok=True)
        
        perf_predictor.reset_path(str(eval_predictor_path / "perf"))
        cost_predictor.reset_path(str(eval_predictor_path / "cost"))
        
        perf_predictor.save(verbose=True)
        cost_predictor.save(verbose=True)
    except Exception as e:
        print(f"Warning: Failed to load or save predictors: {e}")
        perf_predictor = None
        cost_predictor = None

    # Evaluate each fold's model
    for fold in range(k_folds):
        print(f"\n=== Evaluating Fold {fold + 1}/{k_folds} ===")

        # Initialize model
        model = get_2d_model(
            {
                "type": config.model.type,
                "task": config.model.task,
                "num_classes": num_classes,
            }
        )

        # Load checkpoint
        checkpoint_path = (
            Path(quicktune_output_dir)
            / str(config_id)
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

        # Make predictions if predictors are available
        if perf_predictor is not None and cost_predictor is not None:
            try:
                # Prepare test data for prediction
                test_features = pd.DataFrame(dataset_dict["test_data"])
                
                # Make predictions
                predicted_perf = perf_predictor.predict(X=test_features)
                predicted_cost = cost_predictor.predict(X=test_features)
                
                # Store mean predictions
                predicted_performances.append(predicted_perf.mean())
                predicted_costs.append(predicted_cost.mean())
                
                print(f"Predicted Performance: {predicted_performances[-1]:.2f}%")
                print(f"Predicted Cost: {predicted_costs[-1]:.2f}")
            except Exception as e:
                print(f"Warning: Failed to make predictions: {e}")

        # Evaluate model
        fold_metrics = evaluate_model(model, test_loader, criterion, device)
        all_fold_metrics.append(fold_metrics)

        # Print fold results
        print_evaluation_results(fold_metrics, num_classes, fold)

    # Calculate and return average metrics
    avg_metrics = calculate_average_metrics(all_fold_metrics)
    print("\n=== Average Results Across All Folds ===")
    print_evaluation_results(avg_metrics, num_classes)

    # Add predictions to metrics if available
    if predicted_performances and predicted_costs:
        avg_metrics['predicted_performance'] = np.mean(predicted_performances)
        avg_metrics['predicted_cost'] = np.mean(predicted_costs)

    return avg_metrics, num_classes


def main(config: DictConfig) -> None:
    """
    Main entry point for evaluating Quicktune optimization results.

    Args:
        config (DictConfig): Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(config.seed)

    # Get Quicktune output directory from config
    quicktune_output_dir = os.path.join(config.experiment_base_dir, "QuickTune_output")

    # Create directory for evaluation results on the test set
    test_dir = Path(config.experiment_base_dir) / "evaluation_results"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Open text file for console output
    with open(test_dir / "evaluation_output.txt", "w") as f:
        with redirect_stdout(f):
            # Get the best hyperparameters and config ID
            best_hyperparameters, config_id, score, cost = parse_quicktune_results(quicktune_output_dir)

            # Run evaluation on the test set
            avg_metrics, num_classes = test_run_pipeline(
                _pipeline_directory=str(test_dir),
                _previous_pipeline_directory=None,
                config=config,
                quicktune_output_dir=quicktune_output_dir,
                config_id=config_id,
                k_folds=config.data.k_folds,
                **best_hyperparameters,
            )

    # Save results and visualizations
    save_evaluation_results(avg_metrics, test_dir, num_classes)

    # Analyze generalization across all configurations
    # analyze_training_validation_metrics(quicktune_output_dir, config.data.k_folds)

    # Analyze validation-test generalization
    # analyze_validation_test_generalization(quicktune_output_dir, avg_metrics, config.data.k_folds)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
