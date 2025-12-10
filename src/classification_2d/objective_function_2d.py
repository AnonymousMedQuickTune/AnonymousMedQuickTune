import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.classification_2d.models_2d import get_2d_model
from src.classification_2d.preprocess_data_2d import (
    get_brain_tumor_kfold_loaders, load_brain_tumor_dataset)
from src.utils.common_utils import set_seed
from src.utils.logging_utils import (initialize_logging_files, log_gradients,
                                     log_initial_state, log_learning_rate,
                                     log_metrics, log_resources, log_timing,
                                     log_validation_images)
from src.utils.model_lifecycle_utils import (CheckpointManager,
                                             adjust_learning_rate,
                                             evaluate_and_log_metrics,
                                             get_optimizer,
                                             get_warmup_scheduler, set_dropout,
                                             train_epoch)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings(
    "ignore", message="Default grid_sample and affine_grid behavior"
)
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")


def run_2d_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    experimental_setting,
    dataset_dict,
    num_classes,
    **hyperparameters,
):
    """
    Main training pipeline for 2D data model optimization using NePS with K-Fold Cross Validation.

    NOTE: The argument order and parameter names must strictly follow NePS conventions
    to ensure proper optimization and checkpointing functionality.

    Args:
        pipeline_directory (str): Directory where current pipeline results will be saved
        previous_pipeline_directory (str): Directory containing previous pipeline runs
        experimental_setting (DictConfig): Hydra configuration object
        dataset_dict (dict, optional): Combined train+val data and labels dictionary if preloaded
        num_classes (int, optional): Number of classes in the dataset if preloaded
        **hyperparameters: Configuration dictionary containing hyperparameters

    Returns:
        dict: Dictionary containing:
            - objective_to_minimize (float): Negative mean of selected metric (K-fold avg) for NePS optimization
            - cost (float): Cost of the pipeline (optional)
            - extra (dict): Dictionary containing:
                - selected_metric (float): Mean of selected metric (K-fold avg)
                - all_folds_final_metrics (dict): Dictionary containing the mean value for each metric across all folds
    """
    # Set seed for pipeline reproducibility
    set_seed(experimental_setting.seed)

    # Set device (GPU/CPU) for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and move it to the appropriate device
    # CRITICAL: Ensure model type is consistent between training and evaluation
    if "model" in hyperparameters:
        model_type = hyperparameters["model"]  # For QuickTune
        # Synchronize experimental_setting to ensure consistency during evaluation
        experimental_setting.model.type = model_type
        print(f"\nQuickTune selected model: {model_type}\n")
        print(f"[Model Type Sync] Synchronized experimental_setting.model.type to: {model_type}\n")
    else:
        model_type = experimental_setting.model.type  # For NePS
        print(f"\nNePS selected model: {model_type}\n")
    model = get_2d_model(
        {
            "type": model_type,
            "task": experimental_setting.model.task,
            "num_classes": num_classes,
        }
    ).to(device)

    # Get k-fold parameter from experimental_setting or default to 5
    cv_inner_folds = experimental_setting.cv_inner_folds if hasattr(experimental_setting, "cv_inner_folds") else 5

    # Initialize metrics storage for all folds
    all_folds_final_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(pipeline_directory, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # Initialize normalization parameters
    if "autonorm" in str(experimental_setting.pipeline_space):
        # Use normalization stats from NePS hyperparameters
        print(f"\nNormalization parameters from NePS:")
        mean_values = np.array(
            [
                float(hyperparameters["mean_1"]),
                float(hyperparameters["mean_2"]),
                float(hyperparameters["mean_3"]),
            ],
            dtype=np.float32,
        )
        std_values = np.array(
            [
                float(hyperparameters["std_1"]),
                float(hyperparameters["std_2"]),
                float(hyperparameters["std_3"]),
            ],
            dtype=np.float32,
        )
        print(f"Mean: {mean_values}")
        print(f"Std: {std_values}\n")
        normalization_stats = {"mean": mean_values, "std": std_values}
    else:
        # For k-fold CV, normalization stats will be calculated per fold
        normalization_stats = None

    # Run k-fold cross validation
    for fold in range(cv_inner_folds):
        print(f"\nTraining Fold {fold + 1}/{cv_inner_folds}")

        # Create fold-specific directory
        fold_directory = os.path.join(pipeline_directory, f"fold_{fold}")
        os.makedirs(fold_directory, exist_ok=True)

        # Initialize logging files for this fold
        logging_dir = os.path.join(fold_directory, "logging")
        log_files = initialize_logging_files(logging_dir)

        # Get data loaders for this fold
        train_loader, val_loader = get_brain_tumor_kfold_loaders(
            data=dataset_dict["train_val_images"],
            labels=dataset_dict["train_val_labels"],
            cv_inner_folds=cv_inner_folds,
            batch_size=hyperparameters.get("batch_size", 32),
            num_workers=experimental_setting.data.num_workers,
            fold_idx=fold,
            normalization_stats=normalization_stats,  # Will be None if not using autonorm
            augmentation_type=experimental_setting.data.augmentation_type,
        )

        # Apply dropout rate to all applicable layers in the model
        model.apply(lambda m: set_dropout(m, hyperparameters.get("dropout_rate", 0.0)))
        print(f"Model initialized: {model_type}\n")

        # Setup loss function with optional label smoothing
        criterion = nn.CrossEntropyLoss(
            label_smoothing=hyperparameters.get("label_smoothing", 0.0)
        )

        # Initialize optimizer with specified parameters
        optimizer = get_optimizer(
            model=model,
            optimizer_type=hyperparameters.get("optimizer_type", "adam"),
            learning_rate=hyperparameters.get("learning_rate", 1e-3),
            weight_decay=hyperparameters.get("weight_decay", 0.0),
        )

        # Setup warmup scheduler if warmup epochs > 0
        warmup_epochs = hyperparameters.get("warmup_epochs", 0)
        scheduler = (
            get_warmup_scheduler(
                optimizer,
                warmup_epochs,
                len(train_loader),
                hyperparameters.get("learning_rate", 1e-3),
            )
            if warmup_epochs > 0
            else None
        )

        # Initialize metrics dynamically based on all_folds_final_metrics
        base_metrics = list(all_folds_final_metrics.keys()) + ["loss"]
        metrics = {
            "train": {metric: [] for metric in base_metrics},
            "val": {
                **{metric: [] for metric in base_metrics},
                "confusion_matrices": [],  # Additional metric specific to validation
            },
        }

        # Training setup
        # For multi-fidelity compatible searchers (e.g., PriorBand, HyperBand):
        # 'number_of_epochs' is a fidelity parameter dynamically adjusted by NePS.
        # Early optimization runs use fewer epochs for rapid exploration,
        # while promising hyperparameter configurations get more epochs later.
        #
        # For random search:
        # The maximum number of epochs from the pipeline space is used for all evaluations.
        epochs = hyperparameters.get("number_of_epochs")
        if epochs is None and experimental_setting.searcher == "random_search":
            # Load the pipeline space config to get the upper value
            with open(experimental_setting.pipeline_space, "r") as f:
                pipeline_config = yaml.safe_load(f)
                epochs = pipeline_config["number_of_epochs"]["upper"]
            print(f"Random Search: Using maximum epochs value: {epochs}")
        elif epochs is None:
            raise ValueError(
                "number_of_epochs cannot be None for non-random search optimizers"
            )

        # Initialize training components
        checkpoint_manager = CheckpointManager(
            fold_directory, previous_pipeline_directory
        )

        # Modified scaler initialization to be more robust
        scaler = torch.amp.GradScaler() if device == "cuda" else None

        # Load checkpoint and initialize training state
        start_epoch = checkpoint_manager.initialize_training(
            model, optimizer, scheduler, metrics
        )

        # Setup logging
        model.log_gradients = lambda epoch: log_gradients(
            model, epoch, log_files["gradients"]
        )
        log_initial_state(
            log_files=log_files,
            hyperparameters={
                "optimizer_type": hyperparameters.get("optimizer_type", "adam"),
                **hyperparameters,  # Include all other hyperparameters
            },
            experimental_setting=experimental_setting,
            model=model,
            epochs=epochs,
            pipeline_dir=fold_directory,
            prev_pipeline_dir=previous_pipeline_directory,
        )

        # Main training loop
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()

            # Training phase
            train_start_time = time.time()
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                metrics,
                epoch,
                hyperparameters.get("mixup_alpha", 0.0),
            )
            train_time = time.time() - train_start_time

            # Validation phase
            eval_start_time = time.time()
            val_metrics = None  # Initialize val_metrics as None
            if (epoch + 1) % experimental_setting.logging.eval_every == 0 or epoch == epochs - 1:
                val_metrics = evaluate_and_log_metrics(
                    model,
                    val_loader,
                    criterion,
                    device,
                    metrics,
                    phase="val",
                    epoch=epoch,
                )
            eval_time = time.time() - eval_start_time

            # Calculate total epoch time
            epoch_time = time.time() - epoch_start_time

            # Log all metrics and information at the end of the epoch
            log_timing(log_files["timing"], epoch, train_time, eval_time, epoch_time)
            log_learning_rate(log_files["lr"], epoch, optimizer)
            log_resources(log_files["resource"], epoch)

            # Log training metrics
            log_metrics(log_files["metrics"], epoch, "train", train_metrics)

            # Log validation metrics if available
            if val_metrics is not None:
                log_metrics(log_files["metrics"], epoch, "val", val_metrics)

            # Save progress
            checkpoint_manager.save(
                model,
                optimizer,
                scheduler,
                (
                    val_metrics["accuracy"] if val_metrics is not None else 0.0
                ),  # Default to 0.0 if no validation
                experimental_setting,
                num_classes,
                hyperparameters,
                device,
                epoch,
                metrics,
            )

            # Store final metrics for all folds
            if epoch == epochs - 1:
                all_folds_final_metrics["accuracy"].append(val_metrics["accuracy"])
                all_folds_final_metrics["precision"].append(
                    np.mean(val_metrics["precision"]) * 100
                )
                all_folds_final_metrics["recall"].append(
                    np.mean(val_metrics["recall"]) * 100
                )
                all_folds_final_metrics["f1"].append(np.mean(val_metrics["f1"]) * 100)

            # Log metrics to TensorBoard
            writer.add_scalar(f"Loss/train/fold_{fold}", train_metrics["loss"], epoch)
            writer.add_scalar(
                f"Accuracy/train/fold_{fold}", train_metrics["accuracy"], epoch
            )
            writer.add_scalar(
                f"Precision/train/fold_{fold}",
                np.mean(train_metrics["precision"]),
                epoch,
            )
            writer.add_scalar(
                f"Recall/train/fold_{fold}", np.mean(train_metrics["recall"]), epoch
            )
            writer.add_scalar(
                f"F1/train/fold_{fold}", np.mean(train_metrics["f1"]), epoch
            )

            # Log learning rate (moved outside the val_metrics check)
            writer.add_scalar(
                f"Learning_Rate/fold_{fold}", optimizer.param_groups[0]["lr"], epoch
            )

            if val_metrics is not None:
                writer.add_scalar(f"Loss/val/fold_{fold}", val_metrics["loss"], epoch)
                writer.add_scalar(
                    f"Accuracy/val/fold_{fold}", val_metrics["accuracy"], epoch
                )
                writer.add_scalar(
                    f"Precision/val/fold_{fold}",
                    np.mean(val_metrics["precision"]),
                    epoch,
                )
                writer.add_scalar(
                    f"Recall/val/fold_{fold}", np.mean(val_metrics["recall"]), epoch
                )
                writer.add_scalar(
                    f"F1/val/fold_{fold}", np.mean(val_metrics["f1"]), epoch
                )

                # Add confusion matrix as image
                if "confusion_matrices" in val_metrics:
                    fig = plt.figure(figsize=(8, 8))
                    plt.imshow(val_metrics["confusion_matrices"][-1], cmap="Blues")
                    plt.colorbar()
                    plt.title(f"Confusion Matrix - Epoch {epoch}")
                    writer.add_figure(f"Confusion_Matrix/fold_{fold}", fig, epoch)
                    plt.close()

            # Log sample images with predictions (every N epochs or at the end)
            if (
                epoch + 1
            ) % experimental_setting.logging.viz_images_every == 0 or epoch == epochs - 1:
                log_validation_images(writer, model, val_loader, device, fold, epoch)

            # Apply learning rate scheduler after training
            adjust_learning_rate(scheduler)

        print("\nTraining completed!")

    # Close TensorBoard writer
    writer.close()

    # Get the specified metric from final metrics for NePS
    selected_metric = np.mean(all_folds_final_metrics[experimental_setting.metric])
    print(f"\nSelected metric ({experimental_setting.metric}): {selected_metric:.2f}%\n")

    # Convert to NePS loss (negative because NePS minimizes)
    neps_loss = -selected_metric / 100.0  # Normalize to [0,1] range

    # TODO: Add cost calculation (optional)
    cost = 0

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
