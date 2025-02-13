"""
Training module for automated hyperparameter optimization of medical image analysis models.
"""

import logging
import os
import pickle
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from neps import run
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
import warnings

from src.data import get_data_loaders, get_kfold_loaders, load_dataset, WORCDataset
from src.util_functions import (CheckpointManager, adjust_learning_rate,
                                evaluate_and_log_metrics, get_model,
                                get_optimizer, get_warmup_scheduler,
                                initialize_logging_files, log_gradients,
                                log_initial_state, log_learning_rate,
                                log_metrics, log_resources, log_timing,
                                set_dropout, set_seed, train_epoch,
                                yaml_to_neps_pipeline_space)

# TODO: Fix warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior")
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")


def run_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    config,
    dataset_dict,
    num_classes,
    **hyperparameters,
):
    """
    Main training pipeline for model optimization using NePS with K-Fold Cross Validation.

    IMPORTANT: The argument order and parameter names must be as shown for NePS compatibility:
    1. pipeline_directory
    2. previous_pipeline_directory
    3. config
    4. **hyperparameters

    NePS requires these specific positional arguments in this order to manage
    the optimization process and handle checkpointing correctly.

    Args:
        pipeline_directory (str): Directory where current pipeline results will be saved
        previous_pipeline_directory (str): Directory containing previous pipeline runs
        config (DictConfig): Hydra configuration object
        dataset_dict (dict): Dictionary containing all data and labels
        num_classes (int): Number of classes in the dataset
        **hyperparameters: Configuration dictionary containing hyperparameters
        
    Returns:
        dict: Dictionary containing the negative mean of the selected metric as loss for NePS
    """
    # Set seed for pipeline reproducibility
    set_seed(config.seed)
    
    # Get k-fold parameter from config or default to 5
    k_folds = config.get('k_folds', 5)
    
    # Initialize metrics storage for all folds
    all_fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    all_folds_accuracy = []
    all_folds_f1 = []
    all_folds_precision = []
    all_folds_recall = []
    
    # Run k-fold cross validation
    for fold in range(k_folds):
        print(f"\nTraining Fold {fold + 1}/{k_folds}")
        
        # Create fold-specific directory
        fold_directory = os.path.join(pipeline_directory, f"fold_{fold}")
        os.makedirs(fold_directory, exist_ok=True)
        
        # Initialize logging files for this fold
        logging_dir = os.path.join(fold_directory, "logging")
        log_files = initialize_logging_files(logging_dir)
        
        # Get data loaders for this fold
        train_loader, val_loader = get_kfold_loaders(
            dataset_dict['train_data'],
            dataset_dict['train_labels'],
            k_folds=k_folds,
            batch_size=hyperparameters.get("batch_size", 32),
            num_workers=config.data.num_workers,
            fold_idx=fold
        )
        
        # Setup model and training components for this fold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model({
            "type": config.model.type,
            "task": config.model.task,
            "num_classes": num_classes,
        }).to(device)
        model.apply(lambda m: set_dropout(m, hyperparameters.get("dropout_rate", 0.0)))
        print(f"Model initialized: {config.model.type}\n")

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

        # Initialize metrics
        metrics = {
            "train": {
                "loss": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
            },
            "val": {
                "loss": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "confusion_matrices": [],
            },
        }

        # Training setup
        # 'number_of_epochs' is a fidelity parameter dynamically adjusted by NePS.
        # Early optimization runs use fewer epochs for rapid exploration,
        # while promising hyperparameter configurations get more epochs later.
        epochs = hyperparameters["number_of_epochs"]

        # Initialize training components
        checkpoint_manager = CheckpointManager(
            fold_directory, previous_pipeline_directory
        )
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

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
            hyperparameters=hyperparameters,
            config=config,
            model=model,
            epochs=epochs,
            pipeline_dir=fold_directory,
            prev_pipeline_dir=previous_pipeline_directory,
        )

        # Main training loop
        for epoch in range(start_epoch, epochs):
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
                epoch,
                hyperparameters["mixup_alpha"],
            )
            train_time = time.time() - train_start_time

            # Validation phase
            eval_start_time = time.time()
            val_metrics = None  # Initialize val_metrics as None
            if (epoch + 1) % config.logging.eval_every == 0 or epoch == epochs - 1:
                val_metrics = evaluate_and_log_metrics(
                    model, val_loader, criterion, device, metrics, phase="val", epoch=epoch
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
                config,
                num_classes,
                hyperparameters,
                device,
                epoch,
                metrics,
            )

            # After training completion, store the final metrics for this fold
            all_fold_metrics["accuracy"].append(metrics["val"]["accuracy"][-1] if metrics["val"]["accuracy"] else 0)
            all_fold_metrics["precision"].append(np.mean(metrics["val"]["precision"][-1]) * 100 if metrics["val"]["precision"] else 0)
            all_fold_metrics["recall"].append(np.mean(metrics["val"]["recall"][-1]) * 100 if metrics["val"]["recall"] else 0)
            all_fold_metrics["f1"].append(np.mean(metrics["val"]["f1"][-1]) * 100 if metrics["val"]["f1"] else 0)

        print("\nTraining completed!")

        all_folds_accuracy.append(all_fold_metrics["accuracy"][-1])
        print(f"All fold accuracy: {all_folds_accuracy}")
        all_folds_f1.append(all_fold_metrics["f1"][-1])
        print(f"All fold f1: {all_folds_f1}")
        all_folds_precision.append(all_fold_metrics["precision"][-1])
        print(f"All fold precision: {all_folds_precision}")
        all_folds_recall.append(all_fold_metrics["recall"][-1])
        print(f"All fold recall: {all_folds_recall}")

    # Calculate mean metrics across all folds
    final_metrics = {
        'accuracy': np.mean(all_folds_accuracy),
        'precision': np.mean(all_folds_precision),
        'recall': np.mean(all_folds_recall),
        'f1': np.mean(all_folds_f1)
    }
    print(f"\nFinal metrics: {final_metrics}")

    # Print all final metrics
    for metric, value in final_metrics.items():
        print(f"Mean {metric} across {k_folds} folds: {value:.2f}%")

    # Get the specified metric from final metrics for NePS
    selected_metric = final_metrics[config.metric]
    print(f"\nSelected metric ({config.metric}): {selected_metric:.2f}%\n")

    # Convert to NePS loss (negative because NePS minimizes)
    neps_loss = -selected_metric
    return {"loss": neps_loss}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main_experiment_config.yaml",
)
def main(config: DictConfig) -> None:
    """
    Main entry point for the training script.

    Args:
        config (DictConfig): Hydra configuration object
    """
    # Set seed for NePS reproducibility
    set_seed(config.seed)

    # Convert YAML pipeline space configuration into NePS-compatible format
    # NePS requires a specific dictionary structure for hyperparameter definitions
    pipeline_space = yaml_to_neps_pipeline_space(config.pipeline_space)

    # Print configurations
    print("\nconfig: ", config, "\npipeline space: ", pipeline_space, "\n")

    # Save configurations to files
    output_dir = os.path.join(config.experiment_base_dir, "hydra_output")
    os.makedirs(output_dir, exist_ok=True)

    # Load the original pipeline space YAML for compact logging
    with open(config.pipeline_space, "r") as f:
        original_pipeline_space = yaml.safe_load(f)

    for filename, data in [
        ("config.yaml", OmegaConf.to_yaml(config)),
        ("pipeline_space.yaml", yaml.dump(pipeline_space)),  # NePS format
        (
            "pipeline_space_compact.yaml",
            yaml.dump(original_pipeline_space),
        ),  # Original compact format
    ]:
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(data)

    # Try to load from cache first
    data_path = Path(config.data.path)
    cache_file = (
        data_path
        / "cache"
        / f"{config.data.dataset}_bs{pipeline_space['batch_size'].upper}.pkl"
    )
    if cache_file.exists():
        print("\nLoading data from cache...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            dataset_dict = cached_data["dataset_dict"]
            num_classes = cached_data["num_classes"]
    else:
        print(
            "\nNo cache found. Run 'python -m src.preprocess_dataset' first to create cache."
        )
        print("Falling back to regular data loading...")
        # Load the raw dataset first
        dataset = load_dataset(config.data.dataset, data_path=config.data.path)
        
        # Store the training data and labels
        dataset_dict = {
            'train_data': dataset['train_data'],
            'train_labels': dataset['train_labels']
        }
        num_classes = dataset['num_classes']

        # Create initial loaders for printing info
        train_loader = DataLoader(
            WORCDataset(dataset['train_data'], dataset['train_labels']),
            batch_size=pipeline_space["batch_size"].upper,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        val_loader = DataLoader(
            WORCDataset(dataset['val_data'], dataset['val_labels']),
            batch_size=pipeline_space["batch_size"].upper,
            shuffle=False,
            num_workers=config.data.num_workers
        )

    print(f"Dataset '{config.data.dataset}' loaded with {num_classes} classes")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # Run NePS optimization with pre-loaded data
    logging.basicConfig(level=logging.INFO)
    run(
        run_pipeline=lambda pipeline_directory, previous_pipeline_directory, **kwargs: run_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            config=config,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            **kwargs,
        ),
        pipeline_space=pipeline_space,
        root_directory=config.root_directory,
        max_evaluations_total=config.max_evaluations,
        # max_evaluations_per_run=1,
        overwrite_working_directory=False,
        # for debugging:
        # ignore_errors=True,
        # overwrite_working_directory=True if "test" in config.experiment_name else False,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
