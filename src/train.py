"""
Training module for automated hyperparameter optimization of medical image analysis models.
"""

import logging
import os

from torch import nn, optim
import torch
from neps import run
from omegaconf import DictConfig, OmegaConf
import hydra
import yaml

from src.data import get_data_loaders
from src.util_functions import (evaluate_model, get_model, set_seed,
                              yaml_to_neps_pipeline_space, CheckpointManager,
                              train_epoch)


def run_pipeline(
    pipeline_directory, previous_pipeline_directory, config, **hyperparameters
):
    """
    Main training pipeline for model optimization using NePS.
    
    IMPORTANT: The argument order and parameter names must be exactly as shown for NePS compatibility:
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
            - TODO: Add other hyperparameters

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    # Initialize training metrics
    metrics = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': []
    }
    
    # Configure training hyperparameters and directories
    # The number_of_epochs is a fidelity parameter that controls the training duration
    # and is automatically adjusted by NePS during the optimization process
    epochs = hyperparameters["number_of_epochs"]
    print(f"Epoch fidelity: {epochs}")
    print(f"Pipeline directory: {pipeline_directory}")
    print(f"Previous pipeline directory: {previous_pipeline_directory}\n")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(pipeline_directory, previous_pipeline_directory)
    
    # Load previous checkpoint if available
    start_epoch = checkpoint_manager.initialize_training(model, metrics)

    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Main training loop
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)

        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                  scaler, device)
        metrics['train_losses'].append(train_metrics['loss'])
        metrics['train_accuracies'].append(train_metrics['accuracy'])
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")

        # Validation phase
        val_acc = None
        if (epoch + 1) % config.logging.eval_every == 0 or epoch == epochs - 1:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            metrics['val_losses'].append(val_loss)
            metrics['val_accuracies'].append(val_acc)
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Save progress
        checkpoint_manager.save(model, val_acc, config, num_classes, hyperparameters,
                              device, epoch, metrics)

    print("\nTraining completed!")
    final_val_acc = metrics['val_accuracies'][-1] if metrics['val_accuracies'] else 0
    print(f"Final validation accuracy: {final_val_acc:.2f}%\n")

    return {"loss": -final_val_acc}  # NePS minimizes negative accuracy


@hydra.main(
    version_base=None, 
    config_path="../configs/experiments",
    config_name="desmoid_config.yaml"
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
    
    for filename, data in [("config.yaml", OmegaConf.to_yaml(config)), 
                         ("pipeline_space.yaml", yaml.dump(pipeline_space))]:
        with open(os.path.join(output_dir, filename), "w") as f:
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
        overwrite_working_directory=True if "test" in config.experiment_name else False,
    )


if __name__ == "__main__":
    main()
