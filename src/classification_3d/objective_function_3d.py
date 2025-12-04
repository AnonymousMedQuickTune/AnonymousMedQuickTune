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
                                            get_cosine_annealing_scheduler,
                                            train_epoch)
from src.utils.ema_utils import ModelEMA

from src.classification_3d.preprocess_data_3d import (
    get_kfold_dataloaders)
from src.classification_3d.utils.normalization_stats import autonorm
from src.utils.experiment_status_logger import ExperimentStatusLogger
from src.utils.experiment_status_logger import InnerFoldProgressLogger

from src.classification_3d.utils.dataset_info import extract_spatial_size
from src.evaluate_trained_config import evaluate_config_on_validation_set_ensemble

def run_3d_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    experimental_setting,
    dataset_dict=None,
    num_classes=None,
    inner_fold_logger=None,
    use_multifidelity=False,
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
        inner_fold_logger (InnerFoldProgressLogger, optional): Logger for inner fold progress tracking
        use_multifidelity (bool, optional): Whether to use multifidelity optimization
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
    # Extract CV fold from pipeline directory path to use fold-specific seed
    # This ensures inner CV splits are consistent for the same outer fold across different experiments
    pipeline_dir_str = str(pipeline_directory)
    cv_outer_fold = 0  # Default to 0 if not found in path
    if "cv_outer_fold_" in pipeline_dir_str:
        try:
            cv_outer_fold = int(pipeline_dir_str.split("cv_outer_fold_")[-1].split("/")[0])
        except (ValueError, IndexError):
            cv_outer_fold = 0
    
    # Calculate fold-specific seed (same as in run_pipeline)
    # This ensures reproducibility: same outer fold = same seed = same inner CV splits
    fold_specific_seed = experimental_setting.seed + cv_outer_fold
    
    # Set seed for pipeline reproducibility (fold-specific)
    set_seed(fold_specific_seed)

    # Set device (GPU/CPU) for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nHyperparameters: {hyperparameters}\n")  
    
    # Get model type from hyperparameters or experimental_setting
    if "model" in hyperparameters:
        model_type = hyperparameters["model"]  # For QuickTune
        print(f"\nQuickTune selected model: {model_type}\n")

    else:
        model_type = experimental_setting.model.type  # For NePS
        print(f"\nNePS selected model: {model_type}\n")

    # Initialize normalization parameters
    if "autonorm" in str(experimental_setting.pipeline_space):
        # Use normalization stats from NePS hyperparameters
        normalization_stats = autonorm(hyperparameters, is_rgb=False)
    else:
        # For k-fold CV, normalization stats will be calculated per fold > Currently deactivated!
        # NOTE: For biomedical images, normalization is done in the preprocessing step
        normalization_stats = None
    
    # select the dataset_dict based on the selected voxel calculation
    if "voxel_calculation" in str(experimental_setting.pipeline_space):
        if hyperparameters["voxel_calculation"] == "mean":
            dataset = dataset_dict["dataset_dict_mean"]
            voxel_calculation = "mean"  
        elif hyperparameters["voxel_calculation"] == "median":
            dataset = dataset_dict["dataset_dict_median"]
            voxel_calculation = "median"
        elif hyperparameters["voxel_calculation"] == "isotropic":
            dataset = dataset_dict["dataset_dict_isotropic"]
            voxel_calculation = "isotropic"
        elif hyperparameters["voxel_calculation"] == "volumetric_isotropic":
            dataset = dataset_dict["dataset_dict_volumetric_isotropic"]
            voxel_calculation = "volumetric_isotropic"
        else:
            raise ValueError(f"Invalid voxel calculation method: {hyperparameters['voxel_calculation']}")
        voxel_size = dataset["voxel_size"]
    else:
        # Use dataset_dict with median voxel calculation
        if experimental_setting.run_mode == "Baseline":
            dataset = dataset_dict
        else:
            dataset = dataset_dict["dataset_dict_median"]
        voxel_size = dataset["voxel_size"]
        voxel_calculation = "median"

    # Get image size based on developer mode, model type and voxel size
    spatial_size = extract_spatial_size(
        model_type, 
        voxel_calculation, 
        experimental_setting.data.dataset, 
        experimental_setting.developer_mode,
        data_path=experimental_setting.data.path,
        is_medmnist=dataset.get("is_medmnist", False)
    )

    # Calculate total number of inner folds (repeats * splits)
    cv_inner_folds_splits = experimental_setting.cv_inner_folds_splits if hasattr(experimental_setting, "cv_inner_folds_splits") else 5
    cv_inner_folds_repeats = experimental_setting.cv_inner_folds_repeats if hasattr(experimental_setting, "cv_inner_folds_repeats") else 1
    total_inner_folds = cv_inner_folds_repeats * cv_inner_folds_splits

    # Calculate available time per inner fold
    # Formula: cost_to_spend / (max_evaluations * total_inner_folds)
    # This distributes the total time budget across all inner folds of all configs
    # time_per_inner_fold = experimental_setting.cost_to_spend / (experimental_setting.max_evaluations * total_inner_folds)
    budget = 14400  # 4*60*60 > 4 hours budget per config
    time_per_inner_fold = budget / total_inner_folds
    print(f"\nTime budget per inner fold: {time_per_inner_fold:.2f}s ({time_per_inner_fold/3600:.2f}h)\n")  #  (cost_to_spend={experimental_setting.cost_to_spend}s, max_evaluations={experimental_setting.max_evaluations}, total_inner_folds={total_inner_folds})\n")


    all_folds_final_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(pipeline_directory, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # Initialize the inner fold progress logger
    # This logger tracks progress of individual inner folds within each outer fold
    # It automatically extracts the outer fold number and base directory from the pipeline path
    inner_fold_logger = InnerFoldProgressLogger(pipeline_directory)
    
    # Prepare model config (will be reused for each fold)
    model_config = {"type": model_type, "task": experimental_setting.model.task, "num_classes": num_classes}
    
    # Run k-fold cross validation (using RepeatedStratifiedKFold if repeats > 1)
    # Initialize total_cost to accumulate time across all epochs and folds
    total_cost = 0.0
    try:
        for fold in range(total_inner_folds):
            print(f"{'-' * 50}")
            print(f"Training Inner Cross-Validation Fold {fold + 1}/{total_inner_folds} (Repeat {fold // cv_inner_folds_splits + 1}/{cv_inner_folds_repeats}, Split {fold % cv_inner_folds_splits + 1}/{cv_inner_folds_splits})")
            print(f"{'-' * 50}\n")
            
            # Log start of inner fold training
            # This updates the outer fold status file to show this inner fold is now running
            inner_fold_logger.update_inner_fold_progress(
                inner_fold=fold + 1,        # Convert to 1-based indexing (Python uses 0-based, we need 1-based)
                status="in_progress",       # Mark as currently running
                epoch=0,                    # Starting at epoch 0
                total_inner_folds=total_inner_folds   # Total number of inner folds for progress calculation
            )

            # CRITICAL: Set seed before each inner fold to ensure reproducibility
            # Use a combination of outer fold and inner fold to create a unique but deterministic seed
            # This ensures: same outer fold + same inner fold = same seed = same model init + same training
            inner_fold_seed = fold_specific_seed + fold
            set_seed(inner_fold_seed)
            print(f"Setting inner fold seed: {inner_fold_seed} (base: {experimental_setting.seed}, outer_fold: {cv_outer_fold}, inner_fold: {fold})")

            # Store start time for this inner fold (for time limit tracking)
            inner_fold_start_time = time.time()

            # Create fold-specific directory
            fold_directory = os.path.join(pipeline_directory, f"cv_inner_fold_{fold}")
            os.makedirs(fold_directory, exist_ok=True)

            # Initialize logging files for this fold
            logging_dir = os.path.join(fold_directory, "logging")
            log_files = initialize_logging_files(logging_dir)

            # Initialize model for this fold (CRITICAL: after setting seed to ensure deterministic initialization)
            # The model must be re-initialized for each fold to ensure consistent weight initialization
            model = get_3d_model(
                model_config=model_config,
                hyperparameters=hyperparameters,
                developer_mode=experimental_setting.developer_mode,
                spatial_size=spatial_size,
                is_medmnist=dataset.get("is_medmnist", False),
                run_mode=experimental_setting.run_mode
            ).to(device)
            print(f"\nModel initialized for inner fold {fold + 1}: {model_type}\n")
            
            # Get data loaders for this fold
            # Use fold_specific_seed (not inner_fold_seed) for CV splits to ensure same splits across experiments
            # The inner_fold_seed is only for model initialization and training determinism
            train_loader, val_loader = get_kfold_dataloaders(
                seed=fold_specific_seed,
                dataset_name=experimental_setting.data.dataset,
                data=dataset["train_val_images"],
                labels=dataset["train_val_labels"],
                cv_inner_folds_splits=cv_inner_folds_splits,
                cv_inner_folds_repeats=cv_inner_folds_repeats,
                batch_size=hyperparameters.get(
                    "batch_size",
                    getattr(experimental_setting.training, "batch_size", 1)
                ),
                num_workers=experimental_setting.data.num_workers,
                fold_idx=fold,
                voxel_size=dataset["voxel_size"],
                normalization_stats=normalization_stats,
                augmentation_type=experimental_setting.data.augmentation_type,
                developer_mode=experimental_setting.developer_mode,
                spatial_size=spatial_size,
                fold_directory=fold_directory,
                no_validation=experimental_setting.training.no_validation,
                is_medmnist=dataset.get("is_medmnist", False)
            )

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
                weight_decay=hyperparameters.get("weight_decay", 1e-6),
            )

            # Initialize Exponential Moving Average (EMA)
            # ema_decay = hyperparameters.get("ema_decay", 0.999)
            # ema = ModelEMA(model, decay=ema_decay)

            # Training setup: number of epochs (needed for scheduler initialization)
            with open(experimental_setting.pipeline_space, "r") as f:
                pipeline_config = yaml.safe_load(f)

            if "number_of_epochs" in pipeline_config and use_multifidelity:
                # For multi-fidelity runs 'number_of_epochs' is a fidelity parameter dynamically adjusted by NePS.
                # Early optimization runs use fewer epochs for rapid exploration, while promising hyperparameter configurations get more epochs later.
                epochs = hyperparameters['number_of_epochs']
                print(f"\nMulti-fidelity optimization:\n> Using number_of_epochs ({epochs}) as fidelity parameter!\n")
            else:
                # For non-multi-fidelity search spaces (including Baseline runs): Use the number of epochs from the experimental_setting
                epochs = experimental_setting.training.number_of_epochs
                print(f"\nBaseline run or non-multi-fidelity optimization:\nTrain model over {epochs} epochs!")

            # Setup learning rate scheduler
            # Get scheduler type from hyperparameters or experimental_setting (with fallback to "none")
            scheduler_type = hyperparameters.get(
                "scheduler_type", 
                getattr(experimental_setting.training, "scheduler_type", "warmup")
            )  # Options: "warmup" (linear warmup only) or "cosine_warmup" (cosine annealing with optional warmup)
               # Note: warmup_epochs = 0 means no warmup (constant LR for "warmup", pure cosine for "cosine_warmup")
            warmup_epochs = hyperparameters.get("warmup_epochs", 0)
        
            # Get base learning rate for relative cosine_eta_min calculation
            base_lr = hyperparameters.get("learning_rate", 1e-4)
            # cosine_eta_min_factor is a fixed parameter from experimental_setting (not in search space)
            cosine_eta_min_factor = getattr(experimental_setting.training, "cosine_eta_min_factor", 0.01)
            cosine_eta_min = base_lr * cosine_eta_min_factor  # Calculate absolute minimum LR (e.g., 0.01 means eta_min = 0.01 * learning_rate)

            if scheduler_type == "cosine_warmup":
                # Use cosine annealing scheduler (with optional warmup)
                scheduler = get_cosine_annealing_scheduler(
                    optimizer,
                    T_max=epochs,
                    eta_min=cosine_eta_min,
                    warmup_epochs=warmup_epochs
                )
                warmup_str = f" with {warmup_epochs} warmup epochs" if warmup_epochs > 0 else " (no warmup)"
                print(f"Using Cosine Annealing scheduler (T_max={epochs}, eta_min={cosine_eta_min:.2e} = {cosine_eta_min_factor:.3f} * {base_lr:.2e}){warmup_str}")
            else:
                # Use warmup scheduler (or constant LR if warmup_epochs = 0)
                scheduler = get_warmup_scheduler(
                    optimizer,
                    warmup_epochs,
                    len(train_loader),
                    hyperparameters.get("learning_rate", 1e-3),
                )
                if warmup_epochs > 0:
                    print(f"Using Warmup scheduler ({warmup_epochs} epochs)")
                else:
                    print("Using constant learning rate (warmup_epochs = 0)")

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
            best_val_metrics = None  # Store all metrics from the best model (for consistency with test evaluation)
            patience_counter = 0
            patience = experimental_setting.training.patience
            early_stopping_enabled = experimental_setting.training.early_stopping
            use_loss_threshold = experimental_setting.training.use_loss_threshold
            
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
                        total_inner_folds=total_inner_folds       # Total for progress calculation
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
                # Update EMA weights after each training epoch
                # ema.update(model)

                train_time = time.time() - train_start_time
            
                # Validation phase
                eval_start_time = time.time()
                val_metrics = None  # Initialize val_metrics as None
                if val_loader is not None and ((training_epochs + 1) % experimental_setting.logging.eval_every == 0 or training_epochs == epochs - 1):
                    # Evaluate with EMA weights (smoother version of the model)
                    # original_state = model.state_dict()
                    # model.load_state_dict(ema.ema_model.state_dict(), strict=False)

                    val_metrics = evaluate_and_log_metrics(
                        model,
                        val_loader,
                        criterion,
                        device,
                        metrics,
                        phase="val",
                        epoch=training_epochs,
                    )

                    # Restore original weights
                    # model.load_state_dict(original_state, strict=False)

                    # Log EMA validation metric separately to compare
                
                    # if val_metrics is not None:
                    #     writer.add_scalar(
                    #         f"{experimental_setting.metric.upper()}/val_ema/cv_inner_fold_{fold}",
                    #         np.mean(val_metrics[experimental_setting.metric]) * 100,
                    #         training_epochs,
                    #     )

                eval_time = time.time() - eval_start_time

                # Calculate total epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Accumulate epoch time to total_cost
                total_cost += epoch_time

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
                    model,  # ema.ema_model, # Save EMA model instead of raw model
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
                            best_val_metrics = val_metrics.copy()  # Store all metrics from best model
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
                            best_val_metrics = val_metrics.copy()  # Store all metrics from best model
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
                        all_folds_final_metrics["accuracy"].append(val_metrics["accuracy"] * 100)
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
                        all_folds_final_metrics["accuracy"].append(train_metrics["accuracy"] * 100)
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
                    f"Accuracy/train/cv_inner_fold_{fold}", train_metrics["accuracy"] * 100, training_epochs
                )
                writer.add_scalar(
                    f"Precision/train/cv_inner_fold_{fold}",
                    np.mean(train_metrics["precision"]) * 100,
                    training_epochs,
                )
                writer.add_scalar(
                    f"Recall/train/cv_inner_fold_{fold}", np.mean(train_metrics["recall"]) * 100, training_epochs
                )
                writer.add_scalar(
                    f"F1/train/cv_inner_fold_{fold}", np.mean(train_metrics["f1"]) * 100, training_epochs
                )
                writer.add_scalar(
                    f"AUC/train/cv_inner_fold_{fold}", np.mean(train_metrics["auc"]) * 100, training_epochs
                )

                # Log learning rate (moved outside the val_metrics check)
                writer.add_scalar(
                    f"Learning_Rate/cv_inner_fold_{fold}", optimizer.param_groups[0]["lr"], training_epochs
                )

                if val_metrics is not None:
                    writer.add_scalar(f"Loss/val/cv_inner_fold_{fold}", val_metrics["loss"], training_epochs)
                    writer.add_scalar(
                        f"Accuracy/val/cv_inner_fold_{fold}", val_metrics["accuracy"] * 100, training_epochs
                    )
                    writer.add_scalar(
                        f"Precision/val/cv_inner_fold_{fold}",
                        np.mean(val_metrics["precision"]) * 100,
                        training_epochs,
                    )
                    writer.add_scalar(
                        f"Recall/val/cv_inner_fold_{fold}", np.mean(val_metrics["recall"]) * 100, training_epochs
                    )
                    writer.add_scalar(
                        f"F1/val/cv_inner_fold_{fold}", np.mean(val_metrics["f1"]) * 100, training_epochs
                    )
                    writer.add_scalar(
                        f"AUC/val/cv_inner_fold_{fold}", np.mean(val_metrics["auc"]) * 100, training_epochs
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
                
                # Check time limit for this inner fold after completing epoch
                elapsed_time_this_fold = time.time() - inner_fold_start_time
                if elapsed_time_this_fold >= time_per_inner_fold:
                    print(f"\n{'='*80}")
                    print(f"TIME LIMIT REACHED for inner fold {fold + 1}: Stopping training early after epoch {training_epochs + 1}")
                    print(f"Elapsed time for this fold: {elapsed_time_this_fold:.2f}s, Limit: {time_per_inner_fold:.2f}s")
                    print(f"Completed {training_epochs + 1} epochs out of {epochs} planned epochs")
                    print(f"{'='*80}\n")
                    # Break out of training loop for this fold
                    break

            # Store final metrics for all folds (after training is completed, whether by early stopping, time limit, or normal completion)
            # IMPORTANT: For consistency with test evaluation, use metrics from best model if early stopping is enabled
            # Otherwise, use metrics from the last epoch
            if val_metrics is not None:
                # Use validation metrics if available
                if early_stopping_enabled and best_val_metrics is not None:
                    # Use metrics from best model (consistent with test evaluation which loads best_model_checkpoint.pth)
                    final_val_metrics = best_val_metrics
                    print(f"Using metrics from best model (for consistency with test evaluation)")
                elif early_stopping_enabled and best_val_metrics is None:
                    # Edge case: Early stopping enabled but no improvement was ever recorded
                    # This should not happen in normal circumstances, but use last epoch metrics as fallback
                    print(f"Warning: Early stopping enabled but best_val_metrics is None. Using last epoch metrics.")
                    final_val_metrics = val_metrics
                else:
                    # Use metrics from last epoch (no early stopping)
                    final_val_metrics = val_metrics
                
                all_folds_final_metrics["accuracy"].append(final_val_metrics["accuracy"] * 100)
                all_folds_final_metrics["precision"].append(
                    np.mean(final_val_metrics["precision"]) * 100
                )
                all_folds_final_metrics["recall"].append(
                    np.mean(final_val_metrics["recall"]) * 100
                )
                all_folds_final_metrics["f1"].append(np.mean(final_val_metrics["f1"]) * 100)
                all_folds_final_metrics["auc"].append(np.mean(final_val_metrics["auc"]) * 100)
            else:
                # Use training metrics when no validation is available
                all_folds_final_metrics["accuracy"].append(train_metrics["accuracy"] * 100)
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
                    total_inner_folds=total_inner_folds   # Total for progress calculation
                )
        
            print("\nTraining completed!\n")

    except Exception as e:
        # Catch spatial dimension errors (e.g., InstanceNorm with 1x1x1, incompatible model/dataset combinations)
        # Also catch RuntimeError / AcceleratorError (e.g., CUDA errors, OOM errors)
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Flush stdout to ensure error messages appear in log files
        import sys
        sys.stdout.flush()
        
        if "Expected more than 1 spatial element" in error_msg:
            print(f"\n{'='*80}", flush=True)
            print(f"ERROR: Incompatible model/dataset combination detected!", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Model: {model_type}", flush=True)
            print(f"Dataset: {experimental_setting.data.dataset}", flush=True)
            if 'spatial_size' in locals():
                print(f"Spatial size: {spatial_size}", flush=True)
            print(f"Error: {error_msg}", flush=True)
            print(f"\nThis model is not suitable for this input size.", flush=True)
            is_medmnist = dataset.get("is_medmnist", False) if 'dataset' in locals() else False
            if is_medmnist:
                print(f"For MedMNIST datasets (32x32x32), please use DenseNet (with remove_last_block=True), ResNet, or ViT instead.", flush=True)
                print(f"SwinUNETR and EfficientNet are not recommended for small input sizes.", flush=True)
            print(f"{'='*80}\n", flush=True)
            
            # Ensure all_folds_final_metrics is initialized
            if 'all_folds_final_metrics' not in locals():
                all_folds_final_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
            
            # Return worst possible score to signal this configuration is invalid
            # NePS minimizes objective_to_minimize, so we return a large positive value
            # (normal: -selected_metric, so good metric=80% → -80, bad metric=0% → 0)
            # For invalid configs, we return a very large positive value to signal it's bad
            return {
                "objective_to_minimize": 0.0,  # Very large value (NePS minimizes, so this is worst)
                "cost": 0.0,
                "extra": {
                    "selected_metric": 0.0,
                    "all_folds_final_metrics": {
                        metric: 0.0 for metric in all_folds_final_metrics.keys()
                    },
                    "error": "Incompatible model/dataset combination",
                    "error_details": error_msg
                },
            }
        elif "CUDA error" in error_msg or "CUDNN_STATUS_EXECUTION_FAILED" in error_msg:
            # Handle CUDA / accelerator errors gracefully and mark this config as invalid
            print(f"\n{'='*80}", flush=True)
            print(f"ERROR: AcceleratorError during training!", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Model: {model_type}", flush=True)
            print(f"Dataset: {experimental_setting.data.dataset}", flush=True)
            if 'spatial_size' in locals():
                print(f"Spatial size: {spatial_size}", flush=True)
            print(f"Error type: {error_type}", flush=True)
            print(f"Error message: {error_msg}", flush=True)
            print(f"{'='*80}\n", flush=True)

            # Ensure all_folds_final_metrics is initialized
            if 'all_folds_final_metrics' not in locals():
                all_folds_final_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}

            # Return worst possible score to signal this configuration is invalid but keep NePS running
            return {
                "objective_to_minimize": 0.0,
                "cost": 0.0,
                "extra": {
                    "selected_metric": 0.0,
                    "all_folds_final_metrics": {
                        metric: 0.0 for metric in all_folds_final_metrics.keys()
                    },
                    "error": "CUDA error during training",
                    "error_details": error_msg,
                },
            }
        else:
            # Print other exceptions (ValueError or RuntimeError)
            print(f"\n{'='*80}", flush=True)
            print(f"ERROR: {error_type} during training!", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Model: {model_type}", flush=True)
            print(f"Dataset: {experimental_setting.data.dataset}", flush=True)
            if 'spatial_size' in locals():
                print(f"Spatial size: {spatial_size}", flush=True)
            print(f"Error type: {error_type}", flush=True)
            print(f"Error message: {error_msg}", flush=True)
            print(f"{'='*80}\n", flush=True)
            raise
    
    # Close TensorBoard writer
    writer.close()

    # --------------------------------------------------------------------------------------------

    # VALIDATION EVALUATION: Choose between ensemble and mean based on configuration
    # --------------------------------------------------------------------------------------------
    # Note on early stopping compatibility:
    # - early_stopping=False: Uses model_latest_checkpoint.pth (end-of-training models)
    # - early_stopping=True + use_loss_threshold=True: Uses best_model_checkpoint.pth (best loss per fold)
    # - early_stopping=True + use_loss_threshold=False: Uses best_model_checkpoint.pth (best metric per fold)
    # 
    # Important: Early stopping decisions are ALWAYS based on per-fold metrics/loss, not ensemble metrics.
    # This applies to BOTH use_loss_threshold=True (per-fold loss) and use_loss_threshold=False (per-fold metric).
    # The ensemble evaluation uses the best checkpoints from each fold (determined by per-fold early stopping).
    # This works correctly but creates a slight inconsistency: early stopping uses per-fold metrics/loss,
    # while final evaluation uses ensemble metrics. For true ensemble-based early stopping,
    # ensemble evaluation would need to run every epoch (very expensive!).
    # --------------------------------------------------------------------------------------------
    validation_evaluation = getattr(experimental_setting, "validation_evaluation", "mean")
    use_ensemble_validation = (
        validation_evaluation == "ensemble" and 
        cv_inner_folds_repeats > 1
    )
    
    if use_ensemble_validation:
        print(f"\n{'='*80}")
        print(f"USING ENSEMBLE VALIDATION EVALUATION")
        print(f"validation_evaluation={validation_evaluation}, cv_inner_folds_repeats={cv_inner_folds_repeats}")
        print(f"{'='*80}\n")
        
        # Calculate ensemble validation metrics
        # This uses best_model_checkpoint.pth (if early_stopping=True) or model_latest_checkpoint.pth (if early_stopping=False)
        # from each fold, as determined by evaluate_fold() in evaluate_trained_config.py
        ensemble_val_metrics = evaluate_config_on_validation_set_ensemble(
            pipeline_directory=pipeline_directory,
            experimental_setting=experimental_setting,
            dataset=dataset,
            spatial_size=spatial_size,
            num_classes=num_classes,
            hyperparameters=hyperparameters,
            cv_inner_folds_splits=cv_inner_folds_splits,
            cv_inner_folds_repeats=cv_inner_folds_repeats,
            total_inner_folds=total_inner_folds,
            seed=experimental_setting.seed,
            framework="neps"
        )
        
        # Update all_folds_final_metrics with ensemble validation metrics
        all_folds_final_metrics["accuracy"] = [ensemble_val_metrics["accuracy"]]
        all_folds_final_metrics["precision"] = [ensemble_val_metrics["precision"]]
        all_folds_final_metrics["recall"] = [ensemble_val_metrics["recall"]]
        all_folds_final_metrics["f1"] = [ensemble_val_metrics["f1"]]
        all_folds_final_metrics["auc"] = [ensemble_val_metrics["auc"]]
        
        print(f"\nEnsemble validation metrics stored in all_folds_final_metrics")
    else:
        print(f"\n{'='*80}")
        print(f"USING MEAN VALIDATION EVALUATION")
        print(f"validation_evaluation={validation_evaluation}, cv_inner_folds_repeats={cv_inner_folds_repeats}")
        print(f"(Using average of per-fold validation metrics)")
        print(f"{'='*80}\n")

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
    # total_cost accumulates the time across all epochs and all folds
    cost = total_cost

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
