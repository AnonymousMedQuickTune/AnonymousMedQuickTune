"""
This module provides various helper functions for model initialization,
configuration parsing, and evaluation metrics calculation.
"""

import os
import random

import neps
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torchvision import models


def yaml_to_neps_pipeline_space(yaml_path):
    """
    Parse YAML configuration file and convert to NePS pipeline space format.
    Supports both configurations with and without user priors.

    Args:
        yaml_path (str): Path to the YAML configuration file

    Returns:
        dict: NePS-compatible pipeline space dictionary

    Raises:
        ValueError: If unknown parameter type is encountered
    """
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline_space = {}

    # Check if we're using priorband (with user priors) or not
    using_priorband = any(
        "default" in param and "default_confidence" in param
        for param in config.values()
        if isinstance(param, dict)
    )

    for key, value in config.items():
        # Skip non-hyperparameter entries
        if not isinstance(value, dict) or "type" not in value:
            print(f"Skipping non-hyperparameter '{key}': {value}")
            continue

        param_type = value.get("type")
        is_fidelity = value.get("is_fidelity", False)

        # Base parameters for all types
        param_kwargs = {}
        if "lower" in value:
            param_kwargs["lower"] = value["lower"]
        if "upper" in value:
            param_kwargs["upper"] = value["upper"]

        # Handle user priors if present and using priorband
        if using_priorband and "default" in value and "default_confidence" in value:
            param_kwargs.update(
                {
                    "default": value["default"],
                    "default_confidence": value["default_confidence"],
                }
            )

        # Parameter-specific configuration
        if param_type == "float":
            param_kwargs["log"] = value.get("log", False)
            pipeline_space[key] = neps.Float(**param_kwargs)
        elif param_type == "int":
            if is_fidelity:
                param_kwargs["is_fidelity"] = True
            pipeline_space[key] = neps.Integer(**param_kwargs)
        elif param_type == "categorical":
            param_kwargs.pop("lower", None)
            param_kwargs.pop("upper", None)
            param_kwargs["choices"] = value.get("choices")
            pipeline_space[key] = neps.Categorical(**param_kwargs)
        else:
            raise ValueError(f"Unknown type '{param_type}' for parameter '{key}'")

    # Log the configuration mode
    print(f"Configuration mode: {'with' if using_priorband else 'without'} user priors")

    return pipeline_space


def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed (int): Random seed value

    Returns:
        None
    """
    random.seed(seed)  # Python's random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed


def get_model(model_config):
    """
    Create and initialize a model based on the model configuration.

    Args:
        model_config (dict): Model configuration containing:
            - type (str): Type of model ('resnet', 'efficientnet', 'vit',
                         'convnext', 'swin', 'densenet', 'efficientnetv2', 'densenet201')
            - task (str): Type of task ('classification', etc.)
            - num_classes (int): Number of output classes

    Returns:
        nn.Module: Initialized PyTorch model
    """
    model_type = model_config["type"]
    num_classes = model_config["num_classes"]

    # Modern, widely used architectures
    if model_type == "vit":  # Vision Transformer - State of the art
        model = models.vit_b_16(pretrained=True)
        model.heads = nn.Linear(model.hidden_dim, num_classes)
    elif model_type == "convnext":  # Modern CNN architecture
        model = models.convnext_base(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_type == "resnet":  # Classic, reliable architecture
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "swin":  # Modern hierarchical ViT
        model = models.swin_v2_b(pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_type == "efficientnet":  # Efficient modern CNN
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "efficientnetv2":  # Updated EfficientNet
        model = models.efficientnet_v2_s(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "densenet":  # Older architecture
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_type == "densenet201":  # Larger DenseNet variant
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unknown model type: " + model_type)

    return model


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
        for inputs, targets in data_loader:
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
    accuracy = 100.0 * np.mean(all_predictions == all_targets)
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


class CheckpointManager:
    """
    Manages model checkpoints including loading, saving and resuming training.
    """

    def __init__(self, pipeline_directory, previous_pipeline_directory):
        """
        Initialize CheckpointManager.

        Args:
            pipeline_directory (str): Directory for saving new checkpoints
            previous_pipeline_directory (str): Directory containing previous checkpoints
        """
        self.pipeline_directory = pipeline_directory
        self.previous_pipeline_directory = previous_pipeline_directory
        self.checkpoint_name = "model_latest_checkpoint.pth"

    def initialize_training(self, model, metrics):
        """
        Initialize training by loading previous checkpoint if available.

        Args:
            model (nn.Module): Model to load weights into
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
                start_epoch = self._load_checkpoint(checkpoint_path, model, metrics)
                print(f"\nResuming training from epoch {start_epoch}")
            else:
                print("\nNo checkpoint found, starting from epoch 1")
        else:
            print("\nNo previous pipeline directory provided, starting from epoch 1")

        return start_epoch

    def _load_checkpoint(self, checkpoint_path, model, metrics):
        """Internal method to load checkpoint file."""
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics.update(checkpoint["metrics"])
        return checkpoint["epoch"]

    def save(
        self,
        model,
        val_acc,
        config,
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
            val_acc (float): Current validation accuracy
            config (DictConfig): Configuration object
            num_classes (int): Number of classes
            hyperparameters (dict): Training hyperparameters
            device (str): Training device
            epoch (int): Current epoch
            metrics (dict): Training metrics
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "model_type": config.model.type,
            "num_classes": num_classes,
            "hyperparameters": hyperparameters,
            "device": str(device),
            "epoch": epoch + 1,
            "metrics": metrics,
        }

        # Save latest checkpoint (overwrite)
        latest_path = os.path.join(self.pipeline_directory, self.checkpoint_name)
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        if (epoch + 1) % config.logging.save_every == 0:
            periodic_path = os.path.join(
                self.pipeline_directory, f"model_checkpoint_after_{epoch+1}epochs.pth"
            )
            torch.save(checkpoint, periodic_path)


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


def get_warmup_scheduler(optimizer, warmup_epochs, steps_per_epoch):
    """
    Creates a learning rate scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs (int): Number of warmup epochs
        steps_per_epoch (int): Number of steps per epoch

    Returns:
        scheduler: Learning rate scheduler
    """

    def lr_lambda(step):
        current_step = step / steps_per_epoch
        if current_step < warmup_epochs:
            return current_step / warmup_epochs
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def adjust_learning_rate(scheduler):
    """
    Adjusts learning rate according to scheduler.

    Args:
        scheduler: Learning rate scheduler or None
    """
    if scheduler is not None:
        scheduler.step()


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    metrics_dict,
    mixup_alpha=None,
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
        mixup_alpha (float, optional): Mixup alpha parameter

    Returns:
        dict: Dictionary containing loss and accuracy for the epoch
    """
    model.train()

    # Training loop
    for inputs, targets in train_loader:
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply mixup if configured
        if mixup_alpha is not None and mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                    outputs, targets_b
                )
        else:
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Evaluate and log metrics after training
    return evaluate_and_log_metrics(
        model, train_loader, criterion, device, metrics_dict, phase="train"
    )


def evaluate_and_log_metrics(
    model, data_loader, criterion, device, metrics_dict, phase="train"
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

    Returns:
        dict: Current evaluation metrics
    """
    # Evaluate model
    current_metrics = evaluate_model(model, data_loader, criterion, device)

    # Update metrics history
    metrics_dict[f"{phase}_losses"].append(current_metrics["loss"])
    metrics_dict[f"{phase}_accuracies"].append(current_metrics["accuracy"])
    metrics_dict[f"{phase}_precision"].append(current_metrics["precision"])
    metrics_dict[f"{phase}_recall"].append(current_metrics["recall"])
    metrics_dict[f"{phase}_f1"].append(current_metrics["f1"])

    if phase == "val":
        metrics_dict["val_confusion_matrices"].append(
            current_metrics["confusion_matrix"]
        )

    # Print metrics
    phase_name = "Train" if phase == "train" else "Val  "
    print(
        f"{phase_name} - Loss: {current_metrics['loss']:.4f}, "
        f"Acc: {current_metrics['accuracy']:.2f}%"
    )
    print(f"      - Mean Precision: {float(np.mean(current_metrics['precision'])):.4f}")
    print(f"      - Mean Recall: {float(np.mean(current_metrics['recall'])):.4f}")
    print(f"      - Mean F1: {float(np.mean(current_metrics['f1'])):.4f}")

    return current_metrics
