import os
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

# TODO: Use MONAI Metrics + clean & refactor


class CheckpointManager:
    """
    Manages model checkpoints including loading, saving and resuming training.
    """

    def __init__(self, pipeline_directory, previous_pipeline_directory):
        """Initialize CheckpointManager."""
        self.pipeline_directory = pipeline_directory
        self.previous_pipeline_directory = previous_pipeline_directory
        self.checkpoint_name = "model_latest_checkpoint.pth"

    def initialize_training(self, model, optimizer, scheduler, metrics):
        """
        Initialize training by loading previous checkpoint if available.

        Args:
            model (nn.Module): Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into
            metrics (dict): Dictionary to store training metrics

        Returns:
            int: Epoch to start training from
        """
        start_epoch = 0

        if self.previous_pipeline_directory is not None:
            checkpoint_path = os.path.join(
                self.previous_pipeline_directory, self.checkpoint_name
            )
            if os.path.exists(checkpoint_path):
                start_epoch = self._load_checkpoint(
                    checkpoint_path, model, optimizer, scheduler, metrics
                )
                print(f"\nResuming training from epoch {start_epoch}")
            else:
                print("\nNo checkpoint found, starting from epoch 1")
        else:
            print("\nNo previous pipeline directory provided, starting from epoch 1")

        return start_epoch

    def _load_checkpoint(self, checkpoint_path, model, optimizer, scheduler, metrics):
        """
        Internal method to load checkpoint file.

        Args:
            checkpoint_path (str): Path to checkpoint file
            model (nn.Module): Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into
            metrics (dict): Dictionary to store training metrics

        Returns:
            int: Next epoch number
        """
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Load model and training states
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load metrics
        metrics.update(checkpoint["metrics"])

        return checkpoint["epoch"]

    def save(
        self,
        model,
        optimizer,
        scheduler,
        val_acc,
        experimental_setting,
        num_classes,
        hyperparameters,
        device,
        epoch,
        metrics,
    ):
        """
        Save model checkpoint.

        Args:
            model (nn.Module): Model to save
            optimizer: Optimizer to save state
            scheduler: Learning rate scheduler to save state
            val_acc (float): Current validation accuracy
            experimental_setting (DictConfig): Configuration object
            num_classes (int): Number of classes
            hyperparameters (dict): Training hyperparameters
            device (str): Training device
            epoch (int): Current epoch
            metrics (dict): Training metrics
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "model_type": experimental_setting.model.type,
            "num_classes": num_classes,
            "hyperparameters": hyperparameters,
            "device": str(device),
            "epoch": epoch + 1,
            "metrics": metrics,
        }

        # Save scheduler state if it exists
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save latest checkpoint (overwrite)
        latest_path = os.path.join(self.pipeline_directory, self.checkpoint_name)
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        if (epoch + 1) % experimental_setting.logging.save_every == 0:
            periodic_path = os.path.join(
                self.pipeline_directory, f"model_checkpoint_after_{epoch+1}epochs.pth"
            )
            torch.save(checkpoint, periodic_path)


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    metrics_dict,
    epoch,
    mixup_alpha=0.0,
):
    """
    Train model for one epoch and return training metrics.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        metrics_dict (dict): Dictionary containing all metrics history
        epoch (int): Current epoch number
        mixup_alpha (float, optional): Mixup alpha parameter

    Returns:
        dict: Dictionary containing loss and accuracy for the epoch
    """
    model.train()

    # Training loop
    for batch in train_loader:
        if isinstance(batch, dict):
            # Batch is a dict for 3D datasets
            inputs = batch.get("image")
            targets = batch.get("label")
        else:
            # Batch is a tuple for 2D datasets
            inputs, targets = batch
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply mixup if configured
        if mixup_alpha > 0:  #  is not None and mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)

            # Forward pass with mixed precision
            with torch.amp.autocast(device.type):
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                    outputs, targets_b
                )
        else:
            # Forward pass with mixed precision
            with torch.amp.autocast(device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        # Optimizer step first
        optimizer.zero_grad()
        # scaler.scale(loss).backward()

        # # Log gradients before optimizer step if the method exists
        # if hasattr(model, "log_gradients"):
        #     model.log_gradients(epoch)

        # # Optimizer step with gradient scaling
        # scaler.step(optimizer)
        # scaler.update()

        # Inside the training loop, modify the backward pass to handle both with and without scaler
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

    print()  # print empty line for better readability in the logging

    # Evaluate and log metrics after training
    return evaluate_and_log_metrics(
        model, train_loader, criterion, device, metrics_dict, phase="train", epoch=epoch
    )


def evaluate_and_log_metrics(
    model, data_loader, criterion, device, metrics_dict, phase="train", epoch=None
):
    """
    Evaluates the model and logs metrics for either training or validation phase.

    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): DataLoader for either training or validation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to run evaluation on
        metrics_dict (dict): Dictionary containing all metrics history
        phase (str): Either "train" or "val"
        epoch (int, optional): Current epoch number

    Returns:
        dict: Current evaluation metrics
    """
    # Evaluate model
    current_metrics = evaluate_model(model, data_loader, criterion, device)

    # Update metrics history
    metrics_dict[phase]["loss"].append(current_metrics["loss"])
    metrics_dict[phase]["accuracy"].append(current_metrics["accuracy"])
    metrics_dict[phase]["precision"].append(current_metrics["precision"])
    metrics_dict[phase]["recall"].append(current_metrics["recall"])
    metrics_dict[phase]["f1"].append(current_metrics["f1"])

    if phase == "val":
        metrics_dict[phase]["confusion_matrices"].append(
            current_metrics["confusion_matrix"]
        )

    # Print metrics
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    phase_name = "Train" if phase == "train" else "Val  "
    epoch_str = f"[Epoch {epoch+1}] " if epoch is not None else ""
    print(
        f"[{timestamp}]{epoch_str}{phase_name} - "
        f"Loss: {current_metrics['loss']:.4f}, "
        f"Acc: {current_metrics['accuracy']:.2f}%, "
        f"Prec: {float(np.mean(current_metrics['precision']))*100:.2f}%, "
        f"Rec: {float(np.mean(current_metrics['recall']))*100:.2f}%, "
        f"F1: {float(np.mean(current_metrics['f1']))*100:.2f}%"
    )

    return current_metrics


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate a model on a given dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate
        data_loader (DataLoader): DataLoader containing the evaluation dataset
        criterion (nn.Module): Loss function
        device (torch.device): Device to run the evaluation on (CPU/GPU)

    Returns:
        dict: Dictionary containing various evaluation metrics:
            - loss (float): Mean loss across all batches
            - accuracy (float): Classification accuracy in percentage
            - precision (list): Precision for each class
            - recall (list): Recall for each class
            - f1 (list): F1 score for each class
            - confusion_matrix (np.array): Confusion matrix
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                # Batch is a dict for 3D datasets
                inputs = batch.get("image")
                targets = batch.get("label")
            else:
                # Batch is a tuple for 2D datasets
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            predictions = outputs.max(1)[1]

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Convert to numpy arrays for faster computation
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate basic metrics
    accuracy = np.mean(all_predictions == all_targets)
    avg_loss = total_loss / len(data_loader)

    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average=None,  # Calculate metrics for each class
        zero_division=0,  # Handle division by zero
    )

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    # Create metrics dictionary
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": conf_matrix.tolist(),
    }

    return metrics


def set_dropout(module, dropout_rate):
    """
    Recursively sets dropout rate for all dropout layers in the model.

    Args:
        module (nn.Module): PyTorch module
        dropout_rate (float): Dropout rate to set
    """
    if isinstance(module, nn.Dropout):
        module.p = dropout_rate
    for child in module.children():
        set_dropout(child, dropout_rate)


def mixup_data(x, target, mixup_alpha=1.0):
    """
    Performs Mixup augmentation on the input data.

    Args:
        x: Input tensor
        target: Target tensor
        mixup_alpha (float): Mixup alpha parameter for beta distribution

    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
            - mixed_x: Mixed input
            - y_a: First target
            - y_b: Second target
            - lam: Lambda value used for mixing
    """
    if mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = target, target[index]
    return mixed_x, y_a, y_b, lam


def get_warmup_scheduler(optimizer, warmup_epochs, steps_per_epoch, base_lr):
    """
    Create a learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer whose learning rate should be scheduled
        warmup_epochs (int): Number of epochs to warm up for
        steps_per_epoch (int): Number of steps per epoch
        base_lr (float): Target learning rate after warmup

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(epoch):
        """Calculate lr_lambda for LambdaLR scheduler."""
        if epoch <= warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def adjust_learning_rate(scheduler):
    """
    Adjusts learning rate according to scheduler.

    Args:
        scheduler: Learning rate scheduler
    """
    if scheduler is not None:
        scheduler.step()


def get_optimizer(model, optimizer_type, learning_rate, weight_decay):
    """
    Create optimizer based on configuration.

    Args:
        model (nn.Module): Model whose parameters to optimize
        optimizer_type (str): Type of optimizer ('adam', 'adamw', 'sgd')
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay factor

    Returns:
        torch.optim.Optimizer: Configured optimizer

    Raises:
        ValueError: If unknown optimizer type is specified
    """
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
