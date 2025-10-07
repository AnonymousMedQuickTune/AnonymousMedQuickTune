import os
import time
import datetime

import numpy as np
import torch
from torch import nn
import yaml

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
    get_kfold_dataloaders)
from src.classification_3d.utils.normalization_stats import autonorm
from src.utils.experiment_status_logger import ExperimentStatusLogger
from src.utils.experiment_status_logger import InnerFoldProgressLogger

def run_3d_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    experimental_setting,
    dataset_dict=None,
    num_classes=None,
    inner_fold_logger=None,
    **hyperparameters,
):
    """
    Main training pipeline for 3D data model optimization using NePS with K-Fold Cross Validation.

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
                - all_folds_final_metrics (dict): Dictionary containing the mean value for each
                  metric across all folds
    """
    # Set seed for pipeline reproducibility
    set_seed(
        experimental_setting.seed
    )  # For more details on the experimental_setting: pls see configs/main_experiment.yaml

    # Set device (GPU/CPU) for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO @Natalia: Is the search space complete? Yes
    # TODO @Both: Discuss search space in a meeting
    print(f"\nHyperparameters: {hyperparameters}\n")  
    
    # Initialize model and move it to the appropriate device
    if "model" in hyperparameters:
        model_type = hyperparameters["model"]  # For QuickTune
        print(f"\nQuickTune selected model: {model_type}\n")

    else:
        model_type = experimental_setting.model.type  # For NePS
        print(f"\nNePS selected model: {model_type}\n")

    model = get_3d_model(
        {
            "type": model_type,  # Use the model type determined above (either from QuickTune or NePS)
            "task": experimental_setting.model.task,
            "num_classes": num_classes,
        }, hyperparameters
    ).to(device)

    # Get k-fold parameter from experimental_setting or default to 5
    cv_inner_folds = experimental_setting.cv_inner_folds if hasattr(experimental_setting, "cv_inner_folds") else 5

    # Initialize metrics storage for all folds # TODO @Natalia: are there any missing metrics? No
    all_folds_final_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(pipeline_directory, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)
    
    # Initialize normalization parameters and select the dataset_dict based on the selected voxel calculation
    if "autonorm" in str(experimental_setting.pipeline_space):
        # Use normalization stats from NePS hyperparameters
        normalization_stats = autonorm(hyperparameters)
        # Use dataset_dict with median voxel calculation
        dataset = dataset_dict["dataset_dict_median"]  # TODO @Diane: Keep an eye on this!
    elif "baseline" in str(experimental_setting.pipeline_space):
        # For k-fold CV, normalization stats will be calculated per fold
        normalization_stats = None
        # Use dataset_dict with median voxel calculation
        dataset = dataset_dict
    else:
        # For k-fold CV, normalization stats will be calculated per fold
        normalization_stats = None
        # Select the dataset_dict based on the voxel calculation hyperparameter
        if hyperparameters["voxel_calculation"] == "mean":
            dataset_dict = dataset_dict["dataset_dict_mean"]
        elif hyperparameters["voxel_calculation"] == "median":
            dataset_dict = dataset_dict["dataset_dict_median"]
        elif hyperparameters["voxel_calculation"] == "isotropic":
            dataset_dict = dataset_dict["dataset_dict_isotropic"]
        elif hyperparameters["voxel_calculation"] == "volumetric_isotropic":
            dataset_dict = dataset_dict["dataset_dict_volumetric_isotropic"]
        else:
            raise ValueError(f"Invalid voxel calculation method: {hyperparameters['voxel_calculation']}")

    # Initialize the inner fold progress logger
    # This logger tracks progress of individual inner folds within each outer fold
    # It automatically extracts the outer fold number and base directory from the pipeline path
    inner_fold_logger = InnerFoldProgressLogger(pipeline_directory)
    
    # Run k-fold cross validation
    for fold in range(cv_inner_folds):
        print(f"{'-' * 50}")
        print(f"Training Inner Cross-Validation Fold {fold + 1}/{cv_inner_folds}")
        print(f"{'-' * 50}\n")
        
        # Log start of inner fold training
        # This updates the outer fold status file to show this inner fold is now running
        inner_fold_logger.update_inner_fold_progress(
            inner_fold=fold + 1,        # Convert to 1-based indexing (Python uses 0-based, we need 1-based)
            status="in_progress",       # Mark as currently running
            epoch=0,                    # Starting at epoch 0
            total_inner_folds=cv_inner_folds   # Total number of inner folds for progress calculation
        )

        # Create fold-specific directory
        fold_directory = os.path.join(pipeline_directory, f"cv_inner_fold_{fold}")
        os.makedirs(fold_directory, exist_ok=True)

        # Initialize logging files for this fold
        logging_dir = os.path.join(fold_directory, "logging")
        log_files = initialize_logging_files(logging_dir)

        # ... Training ...

        # Get data loaders for this fold
        train_loader, val_loader = get_kfold_dataloaders(
            seed=experimental_setting.seed,
            dataset_name=experimental_setting.data.dataset,
            data=dataset_dict["train_val_images"],
            labels=dataset_dict["train_val_labels"],
            cv_inner_folds=cv_inner_folds,
            batch_size=hyperparameters.get("batch_size", 1),
            num_workers=experimental_setting.data.num_workers,
            fold_idx=fold,
            voxel_size=dataset_dict["voxel_size"],
            normalization_stats=normalization_stats,
            augmentation_type=experimental_setting.data.augmentation_type,
            developer_mode=experimental_setting.developer_mode,
        )

        # TODO @Natalia: Do we need this? > dropout happens somewhere else (happens inside the model)
        # Apply dropout rate to all applicable layers in the model
        # model.apply(lambda m: set_dropout(m, hyperparameters.get("dropout_rate", 0.0)))
        
        print(f"Model initialized: {model_type}\n")

        # Setup loss function with optional label smoothing
        criterion = nn.CrossEntropyLoss(
            label_smoothing=hyperparameters.get("label_smoothing", 0.0)
        )

        # Initialize optimizer with specified parameters
        optimizer = get_optimizer(
            model=model,
            optimizer_type=hyperparameters.get("optimizer_type", "adam"),
            # Get learning_rate from hyperparameters if 'learning_rate' exists in the search space,
            # otherwise use default value of 0.001
            learning_rate=hyperparameters.get("learning_rate", 1e-4),
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

        # Initialize early stopping variables
        best_metric = float('-inf')  # Track best validation metric (higher is better)
        best_loss = float('inf')  # Track best validation loss (lower is better)
        patience_counter = 0
        patience = experimental_setting.training.patience
        early_stopping_enabled = experimental_setting.training.early_stopping
        use_loss_threshold = experimental_setting.training.use_loss_threshold

        # Training setup: number of epochs
        with open(experimental_setting.pipeline_space, "r") as f:
            pipeline_config = yaml.safe_load(f)

        if "number_of_epochs" in pipeline_config:  # TODO @Diane: check how to access fidelity parameter properly (low priority)
            if experimental_setting.searcher == "random_search":
                if experimental_setting.developer_mode:
                    epochs = experimental_setting.training.number_of_epochs
                    print(f"\nRandom Search with a multi-fidelity searchspace.\n> Non-multi-fidelity optimization: Train model over {epochs} epochs!\n")
                else:
                    # The maximum number of epochs from the pipeline space is used for all evaluations.
                    epochs = pipeline_config["number_of_epochs"]["upper"]
                    print(f"\nRandom Search with a multi-fidelity searchspace.\n> Non-multi-fidelity optimization: Train model over {epochs} epochs!\n")
            else:
                print(f"\nMulti-fidelity optimization:\n> Using number_of_epochs ({hyperparameters['number_of_epochs']}) as fidelity parameter!\n")
                # For multi-fidelity compatible searchers (e.g., PriorBand, HyperBand):
                # 'number_of_epochs' is a fidelity parameter dynamically adjusted by NePS.
                # Early optimization runs use fewer epochs for rapid exploration,
                # while promising hyperparameter configurations get more epochs later.
                epochs = pipeline_config["number_of_epochs"]
        else:
            # For non-multi-fidelity search spaces: Use the number of epochs from the experimental_setting
            print(f"\nNon-multi-fidelity optimization:\nTrain model over {experimental_setting.training.number_of_epochs} epochs!")
            epochs = experimental_setting.training.number_of_epochs
        
        # TODO @Natalia: What are the number of epochs to train each config for to reproduce results for each dataset? 50-100
            
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
        for training_epochs in range(start_epoch, epochs):
            epoch_start_time = time.time()

            # Update inner fold progress in inner fold logger at the beginning of an epoch
            # This shows the current training epoch in the status file for real-time progress tracking
            if inner_fold_logger is not None:
                inner_fold_logger.update_inner_fold_progress(
                    inner_fold=fold + 1,
                    status="in_progress",           # Still running
                    epoch=training_epochs + 1,      # Current epoch (1-based for display)
                    total_inner_folds=cv_inner_folds       # Total for progress calculation
                )

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
                accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 1),
            )
            train_time = time.time() - train_start_time
            
            # Validation phase
            eval_start_time = time.time()
            val_metrics = None  # Initialize val_metrics as None
            if val_loader is not None and ((training_epochs + 1) % experimental_setting.logging.eval_every == 0 or training_epochs == epochs - 1):
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

            # Save progress (not the best model, just regular checkpoint)
            checkpoint_manager.save(
                model,
                optimizer,
                scheduler,
                (
                    val_metrics[experimental_setting.metric] if val_metrics is not None else train_metrics[experimental_setting.metric]
                ),  # Use training metric if 'no validation' mode is enabled
                experimental_setting,
                num_classes,
                hyperparameters,
                device,
                training_epochs,
                metrics,
                is_best=False,  # This is not the best model
            )
            
            # Early stopping logic
            if early_stopping_enabled and val_metrics is not None:
                improved = False
                
                if use_loss_threshold:
                    # Use loss for early stopping (minimize loss)
                    current_loss = val_metrics["loss"]
                    if current_loss < best_loss:
                        best_loss = current_loss
                        patience_counter = 0
                        improved = True
                        # Save best model checkpoint
                        checkpoint_manager.save(
                            model,
                            optimizer,
                            scheduler,
                            best_loss,  # Use the actual metric we're tracking
                            experimental_setting,
                            num_classes,
                            hyperparameters,
                            device,
                            training_epochs,
                            metrics,
                            is_best=True
                        )
                        print(f"New best loss: {best_loss:.4f}")
                else:
                    # Use metric for early stopping (maximize metric)
                    current_metric = val_metrics[experimental_setting.metric]
                    if current_metric > best_metric:
                        best_metric = current_metric
                        patience_counter = 0
                        improved = True
                        # Save best model checkpoint
                        checkpoint_manager.save(
                            model,
                            optimizer,
                            scheduler,
                            best_metric,  # Use the actual metric we're tracking
                            experimental_setting,
                            num_classes,
                            hyperparameters,
                            device,
                            training_epochs,
                            metrics,
                            is_best=True
                        )
                        print(f"New best {experimental_setting.metric}: {best_metric:.4f}")
                
                if not improved:
                    patience_counter += 1
                    if use_loss_threshold:
                        if patience_counter == 1:
                            print(f"No improvement for {patience_counter} epoch. Best loss: {best_loss:.4f}")
                        else:
                            print(f"No improvement for {patience_counter} epochs. Best loss: {best_loss:.4f}")
                    else:
                        if patience_counter == 1:
                            print(f"No improvement for {patience_counter} epoch. Best {experimental_setting.metric}: {best_metric:.4f}")
                        else:
                            print(f"No improvement for {patience_counter} epochs. Best {experimental_setting.metric}: {best_metric:.4f}")
                
                if patience_counter >= patience:
                    if use_loss_threshold:
                        print(f"Early stopping triggered after {patience_counter} epochs without loss improvement")
                    else:
                        print(f"Early stopping triggered after {patience_counter} epochs without {experimental_setting.metric} improvement")
                    break

            # Store final metrics for all folds (at the end of training or after early stopping)
            if training_epochs == epochs - 1:
                if val_metrics is not None:
                    # Use validation metrics if available
                    all_folds_final_metrics["accuracy"].append(val_metrics["accuracy"])
                    all_folds_final_metrics["precision"].append(
                        np.mean(val_metrics["precision"]) * 100
                    )
                    all_folds_final_metrics["recall"].append(
                        np.mean(val_metrics["recall"]) * 100
                    )
                    all_folds_final_metrics["f1"].append(np.mean(val_metrics["f1"]) * 100)
                    all_folds_final_metrics["auc"].append(np.mean(val_metrics["auc"]) * 100)
                else:
                    # Use training metrics when no validation is available
                    all_folds_final_metrics["accuracy"].append(train_metrics["accuracy"])
                    all_folds_final_metrics["precision"].append(
                        np.mean(train_metrics["precision"]) * 100
                    )
                    all_folds_final_metrics["recall"].append(
                        np.mean(train_metrics["recall"]) * 100
                    )
                    all_folds_final_metrics["f1"].append(np.mean(train_metrics["f1"]) * 100)
                    all_folds_final_metrics["auc"].append(np.mean(train_metrics["auc"]) * 100)

            # Log metrics to TensorBoard
            writer.add_scalar(f"Loss/train/cv_inner_fold_{fold}", train_metrics["loss"], training_epochs)
            writer.add_scalar(
                f"Accuracy/train/cv_inner_fold_{fold}", train_metrics["accuracy"], training_epochs
            )
            writer.add_scalar(
                f"Precision/train/cv_inner_fold_{fold}",
                np.mean(train_metrics["precision"]),
                training_epochs,
            )
            writer.add_scalar(
                f"Recall/train/cv_inner_fold_{fold}", np.mean(train_metrics["recall"]), training_epochs
            )
            writer.add_scalar(
                f"F1/train/cv_inner_fold_{fold}", np.mean(train_metrics["f1"]), training_epochs
            )
            writer.add_scalar(
                f"AUC/train/cv_inner_fold_{fold}", np.mean(train_metrics["auc"]), training_epochs
            )

            # Log learning rate (moved outside the val_metrics check)
            writer.add_scalar(
                f"Learning_Rate/cv_inner_fold_{fold}", optimizer.param_groups[0]["lr"], training_epochs
            )

            if val_metrics is not None:
                writer.add_scalar(f"Loss/val/cv_inner_fold_{fold}", val_metrics["loss"], training_epochs)
                writer.add_scalar(
                    f"Accuracy/val/cv_inner_fold_{fold}", val_metrics["accuracy"], training_epochs
                )
                writer.add_scalar(
                    f"Precision/val/cv_inner_fold_{fold}",
                    np.mean(val_metrics["precision"]),
                    training_epochs,
                )
                writer.add_scalar(
                    f"Recall/val/cv_inner_fold_{fold}", np.mean(val_metrics["recall"]), training_epochs
                )
                writer.add_scalar(
                    f"F1/val/cv_inner_fold_{fold}", np.mean(val_metrics["f1"]), training_epochs
                )
                writer.add_scalar(
                    f"AUC/val/cv_inner_fold_{fold}", np.mean(val_metrics["auc"]), training_epochs
                )

                # Add confusion matrix as image
                if "confusion_matrices" in val_metrics:
                    fig = plt.figure(figsize=(8, 8))
                    plt.imshow(val_metrics["confusion_matrices"][-1], cmap="Blues")
                    plt.colorbar()
                    plt.title(f"Confusion Matrix - Epoch {training_epochs}")
                    writer.add_figure(f"Confusion_Matrix/cv_inner_fold_{fold}", fig, training_epochs)
                    plt.close()

            # Log sample images with predictions (every N epochs or at the end)
            if val_loader is not None and ((training_epochs + 1) % experimental_setting.logging.viz_images_every == 0 or training_epochs == epochs - 1):
                log_validation_images(writer, model, val_loader, device, fold, training_epochs)

            # Apply learning rate scheduler after training
            adjust_learning_rate(scheduler)

        # Store final metrics for all folds (after training is completed, whether by early stopping or normal completion)
        if val_metrics is not None:
            # Use validation metrics if available
            all_folds_final_metrics["accuracy"].append(val_metrics["accuracy"])
            all_folds_final_metrics["precision"].append(
                np.mean(val_metrics["precision"]) * 100
            )
            all_folds_final_metrics["recall"].append(
                np.mean(val_metrics["recall"]) * 100
            )
            all_folds_final_metrics["f1"].append(np.mean(val_metrics["f1"]) * 100)
            all_folds_final_metrics["auc"].append(np.mean(val_metrics["auc"]) * 100)
        else:
            # Use training metrics when no validation is available
            all_folds_final_metrics["accuracy"].append(train_metrics["accuracy"])
            all_folds_final_metrics["precision"].append(
                np.mean(train_metrics["precision"]) * 100
            )
            all_folds_final_metrics["recall"].append(
                np.mean(train_metrics["recall"]) * 100
            )
            all_folds_final_metrics["f1"].append(np.mean(train_metrics["f1"]) * 100)
            all_folds_final_metrics["auc"].append(np.mean(train_metrics["auc"]) * 100)

        # Log completion of inner fold training and mark inner fold as completed.
        if inner_fold_logger is not None:
            inner_fold_logger.update_inner_fold_progress(
                inner_fold=fold + 1,
                status="completed",         # Mark as finished
                total_inner_folds=cv_inner_folds   # Total for progress calculation
            )
        
        print("\nTraining completed!\n")
    
    # Close TensorBoard writer
    writer.close()

    # --------------------------------------------------------------------------------------------

    # For NePS:
    # NePS requires a single objective (loss) to minimize. We use the negative of one selected
    # metric (e.g., f1-score) as the loss. Additional metrics are logged in 'info_dict'.

    # Get the specified metric from final metrics for NePS
    selected_metric = np.mean(all_folds_final_metrics[experimental_setting.metric])
    print(f"\nSelected metric ({experimental_setting.metric}): {selected_metric:.2f}%\n")

    # Convert to NePS loss (negative because NePS minimizes)
    neps_loss = -selected_metric

    # Add cost calculation (optional, currently unused feature in this project)
    # If we want to stop a NePS run after a certain total max_cost_toal is reached, we can define
    # the cost of one config evaluation, e.g. the time it takes to run a k-fold cv on one experimental_setting.
    cost = epoch_time

    # TODO @Natalia: What are the goal performances that need to be achieved to reproduce results for each dataset? 82 for Lipo.
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
