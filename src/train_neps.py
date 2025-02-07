"""
Training module for automated hyperparameter optimization of medical image analysis models.
"""

import logging
import os
import time
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from neps import run
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim

from src.data import get_data_loaders
from src.util_functions import (CheckpointManager, adjust_learning_rate,
                                evaluate_and_log_metrics, get_model,
                                get_optimizer, get_warmup_scheduler,
                                initialize_logging_files, log_gradients,
                                log_initial_state, log_learning_rate,
                                log_metrics, log_resources, log_timing,
                                set_dropout, set_seed, train_epoch,
                                yaml_to_neps_pipeline_space)


def run_pipeline(
    pipeline_directory, 
    previous_pipeline_directory, 
    config, 
    train_loader,
    val_loader,
    num_classes,
    **hyperparameters
):
    """
    Main training pipeline for model optimization using NePS.

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
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_classes (int): Number of classes in the dataset
        **hyperparameters: Configuration dictionary containing hyperparameters:
            - learning_rate (float): Learning rate for optimizer
            - batch_size (int): Batch size for training
            - number_of_epochs (int): Number of training epochs
            - label_smoothing (float): Label smoothing factor
            - optimizer_type (str): Optimizer type
            - weight_decay (float): Weight decay for optimizer
            - dropout_rate (float): Dropout rate for model
            - mixup_alpha (float): Mixup alpha for mixup function
            - warmup_epochs (int): Warmup epochs for warmup scheduler
            - TODO: Add other hyperparameters?

    Returns:
        dict: Dictionary containing the negative validation accuracy as loss for NePS
    """
    # Set seed for pipeline reproducibility
    set_seed(config.seed)

    # Initialize logging files
    logging_dir = os.path.join(pipeline_directory, "logging")
    log_files = initialize_logging_files(logging_dir)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Setup model
    model_config = {
        "type": config.model.type,
        "task": config.model.task,
        "num_classes": num_classes,
    }
    model = get_model(model_config).to(device)
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

    # Select GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize training components
    checkpoint_manager = CheckpointManager(
        pipeline_directory, previous_pipeline_directory
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

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
        pipeline_dir=pipeline_directory,
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
                model, val_loader, criterion, device, metrics, phase="val"
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

    print("\nTraining completed!")

    # Get final metrics
    final_metrics = {
        "accuracy": metrics["val"]["accuracy"][-1] if metrics["val"]["accuracy"] else 0,
        "precision": (
            np.mean(metrics["val"]["precision"][-1])
            if metrics["val"]["precision"]
            else 0
        ),
        "recall": (
            np.mean(metrics["val"]["recall"][-1]) if metrics["val"]["recall"] else 0
        ),
        "f1": np.mean(metrics["val"]["f1"][-1]) if metrics["val"]["f1"] else 0,
    }

    # Get the specified metric from final metrics
    selected_metric = final_metrics[config.metric]
    print(f"Final {config.metric}: {100 * selected_metric:.2f}%\n")

    # Convert to NePS loss (negative because NePS minimizes)
    neps_loss = -100 * selected_metric  # Scale to percentage and negate
    return {"loss": neps_loss}


@hydra.main(
    version_base=None,
    config_path="../configs/experiments",
    config_name="desmoid_config.yaml",
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
    cache_file = data_path / "cache" / f"{config.data.dataset}_bs{pipeline_space['batch_size'].upper}.pkl"
    if cache_file.exists():
        print("\nLoading data from cache...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            train_loader = cached_data["train_loader"]
            val_loader = cached_data["val_loader"]
            num_classes = cached_data["num_classes"]
    else:
        print("\nNo cache found. Run 'python -m src.preprocess_dataset' first to create cache.")
        print("Falling back to regular data loading...")
        train_loader, val_loader, num_classes = get_data_loaders(
            config.data.dataset,
            config.data.num_workers,
            batch_size=pipeline_space["batch_size"].upper,
            split="train",
            data_path=config.data.path,
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
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            **kwargs,
        ),
        pipeline_space=pipeline_space,
        root_directory=config.root_directory,
        max_evaluations_total=config.max_evaluations,
        max_evaluations_per_run=1,
        overwrite_working_directory=False,
        # for debugging:
        ignore_errors=True,
        # overwrite_working_directory=True if "test" in config.experiment_name else False,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
