import os
import time

import numpy as np
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from src.classification_3d.models_3d import get_3d_model 
from src.utils.common_utils import set_seed
from src.utils.logging_utils import (initialize_logging_files, log_gradients,
                                     log_initial_state, log_learning_rate,
                                     log_metrics, log_validation_images, log_resources, log_timing)
from src.utils.model_lifecycle_utils import (CheckpointManager,
                                            adjust_learning_rate,
                                            evaluate_and_log_metrics,
                                            get_optimizer,
                                            get_warmup_scheduler,
                                            train_epoch)
from src.classification_3d.preprocess_data_3d import (
    get_dataloaders)

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
            # How to deal with the model config?
            # "config": config.model.search_space,
        }, hyperparameters
    ).to(device)
    epoch = 10

    # Initialize metrics storage for all folds
    all_folds_final_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(pipeline_directory, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # Example: Logging 5-fold cross validation for NePS
    k_folds = config.data.k_folds if hasattr(config.data, "k_folds") else 5
    for fold in range(k_folds):
        print(f"\nTraining Fold {fold + 1}/{k_folds}\n... training...")

        # Create fold-specific directory
        fold_directory = os.path.join(pipeline_directory, f"fold_{fold}")
        os.makedirs(fold_directory, exist_ok=True)

        # Initialize logging files for this fold
        logging_dir = os.path.join(fold_directory, "logging")
        log_files = initialize_logging_files(logging_dir)

        # ... Training ...

        # Get data loaders for this fold
        train_loader, val_loader = get_dataloaders(
            data=dataset_dict["train_val_data"],
            labels=dataset_dict["train_val_labels"],
            k_folds=k_folds,
            batch_size=hyperparameters.get("batch_size", 32),
            num_workers=config.data.num_workers,
            fold_idx=fold,
            developer_mode=config.developer_mode,
        )

        # Setup loss function with optional label smoothing
        criterion = nn.CrossEntropyLoss(
            label_smoothing=hyperparameters.get("label_smoothing", 0.0)
        )

        optimizer = get_optimizer(
            model=model,
            optimizer_type=hyperparameters.get("optimizer_type", "adam"),
            # Get learning_rate from hyperparameters if 'learning_rate' exists in the search space,
            # otherwise use default value of 0.001
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
            config=config,
            model=model,
            epochs=epoch,
            pipeline_dir=fold_directory,
            prev_pipeline_dir=previous_pipeline_directory,
        )

        # Main training loop
        for training_epochs in range(start_epoch, epoch):
            epoch_start_time = time.time()

            # Apply warmup scheduler at the beginning of each epoch
            adjust_learning_rate(scheduler)

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
                training_epochs,
                hyperparameters.get("mixup_alpha", 0.0),
            )
            train_time = time.time() - train_start_time
            
            # Validation phase
            eval_start_time = time.time()
            val_metrics = None  # Initialize val_metrics as None
            if (training_epochs + 1) % config.logging.eval_every == 0 or training_epochs == epoch - 1:
                val_metrics = evaluate_and_log_metrics(
                    model,
                    val_loader,
                    criterion,
                    device,
                    metrics,
                    phase="val",
                    epoch=training_epochs,
                )
            eval_time = time.time() - eval_start_time

            # Calculate total epoch time
            epoch_time = time.time() - epoch_start_time


            # Log all metrics and information at the end of the epoch
            log_timing(log_files["timing"], training_epochs, train_time, eval_time, epoch_time)
            log_learning_rate(log_files["lr"], training_epochs, optimizer)
            log_resources(log_files["resource"], training_epochs)

            # Log training metrics
            log_metrics(log_files["metrics"], training_epochs, "train", train_metrics)

            # Log validation metrics if available
            if val_metrics is not None:
                log_metrics(log_files["metrics"], training_epochs, "val", val_metrics)

            # Save progress
            checkpoint_manager.save(
                model,
                optimizer,
                scheduler,
                (
                    val_metrics["accuracy"] if val_metrics is not None else 0.0
                ),  # Default to 0.0 if no validation
                config,
                num_classes,
                hyperparameters,
                device,
                training_epochs,
                metrics,
            )
            
            # Store final metrics for all folds
            if training_epochs == epoch - 1:
                all_folds_final_metrics["accuracy"].append(val_metrics["accuracy"])
                all_folds_final_metrics["precision"].append(
                    np.mean(val_metrics["precision"]) * 100
                )
                all_folds_final_metrics["recall"].append(
                    np.mean(val_metrics["recall"]) * 100
                )
                all_folds_final_metrics["f1"].append(np.mean(val_metrics["f1"]) * 100)

            # Log metrics to TensorBoard
            writer.add_scalar(f"Loss/train/fold_{fold}", train_metrics["loss"], training_epochs)
            writer.add_scalar(
                f"Accuracy/train/fold_{fold}", train_metrics["accuracy"], training_epochs
            )
            writer.add_scalar(
                f"Precision/train/fold_{fold}",
                np.mean(train_metrics["precision"]),
                training_epochs,
            )
            writer.add_scalar(
                f"Recall/train/fold_{fold}", np.mean(train_metrics["recall"]), training_epochs
            )
            writer.add_scalar(
                f"F1/train/fold_{fold}", np.mean(train_metrics["f1"]), training_epochs
            )

            # Log learning rate (moved outside the val_metrics check)
            writer.add_scalar(
                f"Learning_Rate/fold_{fold}", optimizer.param_groups[0]["lr"], training_epochs
            )

            if val_metrics is not None:
                writer.add_scalar(f"Loss/val/fold_{fold}", val_metrics["loss"], training_epochs)
                writer.add_scalar(
                    f"Accuracy/val/fold_{fold}", val_metrics["accuracy"], training_epochs
                )
                writer.add_scalar(
                    f"Precision/val/fold_{fold}",
                    np.mean(val_metrics["precision"]),
                    training_epochs,
                )
                writer.add_scalar(
                    f"Recall/val/fold_{fold}", np.mean(val_metrics["recall"]), training_epochs
                )
                writer.add_scalar(
                    f"F1/val/fold_{fold}", np.mean(val_metrics["f1"]), training_epochs
                )

                # Add confusion matrix as image
                if "confusion_matrices" in val_metrics:
                    fig = plt.figure(figsize=(8, 8))
                    plt.imshow(val_metrics["confusion_matrices"][-1], cmap="Blues")
                    plt.colorbar()
                    plt.title(f"Confusion Matrix - Epoch {training_epochs}")
                    writer.add_figure(f"Confusion_Matrix/fold_{fold}", fig, training_epochs)
                    plt.close()

            # Log sample images with predictions (every N epochs or at the end)
            if (
                training_epochs + 1
            ) % config.logging.viz_images_every == 0 or training_epochs == epoch - 1:
                log_validation_images(writer, model, val_loader, device, fold, training_epochs)


        print("\nTraining completed!")
    
    # Close TensorBoard writer
    writer.close()

    # Placeholder for metric values: The metrics from each of the 5 folds in the last epoch
    # Testing?
    #all_folds_final_metrics = {
    #    "accuracy": [90, 87, 85, 89, 88],
    #    "precision": [90, 87, 86, 89, 88],
    #    "recall": [90, 87, 82, 89, 88],
    #    "f1": [90, 87, 83, 89, 88],
    #}
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
