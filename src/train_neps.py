"""
Training module for automated hyperparameter optimization of medical image analysis models.
"""

import logging
import os

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
                                get_warmup_scheduler, set_dropout, set_seed,
                                train_epoch, yaml_to_neps_pipeline_space)


def run_pipeline(
    pipeline_directory, previous_pipeline_directory, config, **hyperparameters
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

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load dataset and create data loaders
    train_loader, val_loader, num_classes = get_data_loaders(
        config.data.dataset,
        config.data.num_workers,
        hyperparameters["batch_size"],
        split="train",
        data_path=config.data.path,
    )
    print(f"Dataset '{config.data.dataset}' loaded with {num_classes} classes")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # Create model config dictionary, get model and move to device
    model_config = {
        "type": config.model.type,
        "task": config.model.task,
        "num_classes": num_classes,
    }
    model = get_model(model_config)
    model = model.to(device)
    print(f"Model initialized: {config.model.type}\n")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=hyperparameters["label_smoothing"])

    # Create optimizer based on type
    if hyperparameters["optimizer_type"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"],
        )
    elif hyperparameters["optimizer_type"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"],
        )
    elif hyperparameters["optimizer_type"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"],
            momentum=0.9,
        )
    else:
        raise ValueError(f"Invalid optimizer type: {hyperparameters['optimizer_type']}")

    # Apply dropout to model
    model.apply(lambda m: set_dropout(m, hyperparameters["dropout_rate"]))

    # Initialize mixup if alpha > 0
    mixup_alpha = hyperparameters["mixup_alpha"]

    # Warmup scheduler
    warmup_epochs = hyperparameters["warmup_epochs"]
    if warmup_epochs > 0:
        scheduler = get_warmup_scheduler(optimizer, warmup_epochs, len(train_loader))
    else:
        raise ValueError("Warmup epochs must be greater than 0")

    # Initialize training metrics
    metrics = {
        "train_losses": [],
        "train_accuracies": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_losses": [],
        "val_accuracies": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_confusion_matrices": [],
    }

    # Configure training hyperparameters and directories
    # The number_of_epochs is a fidelity parameter that controls the training duration
    # and is automatically adjusted by NePS during the optimization process
    epochs = hyperparameters["number_of_epochs"]
    print(f"Epoch fidelity: {epochs}")
    print(f"Pipeline directory: {pipeline_directory}")
    print(f"Previous pipeline directory: {previous_pipeline_directory}\n")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        pipeline_directory, previous_pipeline_directory
    )

    # Load previous checkpoint if available
    start_epoch = checkpoint_manager.initialize_training(model, metrics)

    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create metrics logging directory and file
    # logging_dir = os.path.join(pipeline_directory, "logging")
    # os.makedirs(logging_dir, exist_ok=True)
    logging_file = os.path.join(pipeline_directory, "logging.csv")
    
    # Create CSV header if file doesn't exist
    if not os.path.exists(logging_file):
        with open(logging_file, 'w', encoding='utf-8') as f:
            f.write("epoch,phase,loss,accuracy,precision,recall,f1\n")

    # Main training loop
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)

        # Apply warmup scheduler if configured
        if epoch < warmup_epochs:
            adjust_learning_rate(scheduler)

        # Training phase
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            metrics,
            mixup_alpha,
        )
        
        # Log training metrics
        with open(logging_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1},train,{train_metrics['loss']:.4f},{train_metrics['accuracy']:.4f},"
                   f"{np.mean(train_metrics['precision']):.4f},{np.mean(train_metrics['recall']):.4f},"
                   f"{np.mean(train_metrics['f1']):.4f}\n")

        # Validation phase (based on eval_every or last epoch)
        val_metrics = None
        if (epoch + 1) % config.logging.eval_every == 0 or epoch == epochs - 1:
            val_metrics = evaluate_and_log_metrics(
                model, val_loader, criterion, device, metrics, phase="val"
            )
            
            # Log validation metrics
            with open(logging_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch+1},val,{val_metrics['loss']:.4f},{val_metrics['accuracy']:.4f},"
                       f"{np.mean(val_metrics['precision']):.4f},{np.mean(val_metrics['recall']):.4f},"
                       f"{np.mean(val_metrics['f1']):.4f}\n")

        # Save progress
        checkpoint_manager.save(
            model,
            val_metrics["accuracy"],
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
        "accuracy": metrics["val_accuracies"][-1] if metrics["val_accuracies"] else 0,
        "precision": (
            np.mean(metrics["val_precision"][-1]) if metrics["val_precision"] else 0
        ),
        "recall": np.mean(metrics["val_recall"][-1]) if metrics["val_recall"] else 0,
        "f1": np.mean(metrics["val_f1"][-1]) if metrics["val_f1"] else 0,
    }

    selected_metric = final_metrics[config.metric]
    print(f"Final {config.metric}: {selected_metric:.2f}%\n")

    return {"loss": -selected_metric}  # NePS minimizes negative metric


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

    for filename, data in [
        ("config.yaml", OmegaConf.to_yaml(config)),
        ("pipeline_space.yaml", yaml.dump(pipeline_space)),
    ]:
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(data)

    # Run NePS optimization
    logging.basicConfig(level=logging.INFO)
    run(
        run_pipeline=lambda pipeline_directory, previous_pipeline_directory, **kwargs: run_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            config=config,
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
