import os

import numpy as np
import torch
import yaml


def initialize_logging_files(logging_dir):
    """
    Initialize logging directory and create log files with headers.

    Args:
        logging_dir (str): Directory path where log files should be created

    Returns:
        dict: Dictionary containing paths to all log files
    """
    # Create logging directory
    os.makedirs(logging_dir, exist_ok=True)

    # Define log file paths
    log_files = {
        "metrics": os.path.join(logging_dir, "metrics.csv"),
        "gradients": os.path.join(logging_dir, "gradients.csv"),
        "hyperparameters": os.path.join(logging_dir, "hyperparameters.yaml"),
        "lr": os.path.join(logging_dir, "learning_rates.csv"),
        "resource": os.path.join(logging_dir, "resources.csv"),
        "timing": os.path.join(logging_dir, "timing.csv"),
        "model_info": os.path.join(logging_dir, "model_info.yaml"),
    }

    # Create CSV headers if files don't exist
    headers = {
        "metrics": "epoch,phase,loss,accuracy,precision,recall,f1",
        "gradients": "epoch,layer_name,avg_grad,max_grad",
        "lr": "epoch,learning_rate",
        "resource": "epoch,gpu_memory_allocated,gpu_memory_cached",
        "timing": "epoch,train_time,eval_time,total_time",
    }

    for file_key, header in headers.items():
        if not os.path.exists(log_files[file_key]):
            with open(log_files[file_key], "w", encoding="utf-8") as f:
                f.write(f"{header}\n")

    return log_files


def log_initial_state(
    log_files, hyperparameters, config, model, epochs, pipeline_dir, prev_pipeline_dir
):
    """
    Log model architecture information and print basic run configuration.

    Args:
        log_files (dict): Dictionary containing paths to log files for metrics, gradients,
                         hyperparameters, learning rates, resources, timing, and model info
        hyperparameters (dict): Dictionary of training hyperparameters including optimizer
                               settings, learning rates, and other training parameters
        config: Configuration object containing model architecture and training settings
        model (nn.Module): The initialized PyTorch model to be trained
        epochs (int): Total number of training epochs to run
        pipeline_dir (str): Path to the current pipeline's output directory where
                          checkpoints and logs will be saved
        prev_pipeline_dir (str): Path to previous pipeline directory for resuming
                                training, or None if starting fresh

    Note:
        This function logs the model architecture and hyperparameters to YAML files
        and prints the epoch fidelity and directory paths to the console.
    """
    # Log model architecture info
    log_model_info(log_files["model_info"], model, config, hyperparameters)

    # Print configuration for convenience
    print(f"Epoch fidelity: {epochs}")
    print(f"Pipeline directory: {pipeline_dir}")
    print(f"Previous pipeline directory: {prev_pipeline_dir}\n")


def log_metrics(log_file, epoch, phase, metrics):
    """
    Log training or validation metrics to a CSV file.

    Args:
        log_file (str): Path to the metrics log file
        epoch (int): Current epoch number
        phase (str): Either 'train' or 'val'
        metrics (dict): Dictionary containing the metrics to log:
            - loss (float): Loss value
            - accuracy (float): Accuracy value
            - precision (list): Precision values
            - recall (list): Recall values
            - f1 (list): F1 values
    """
    metrics_line = (
        f"{epoch+1},"
        f"{phase},"
        f"{metrics['loss']:.4f},"
        f"{metrics['accuracy']:.4f},"
        f"{np.mean(metrics['precision'])*100:.4f},"
        f"{np.mean(metrics['recall'])*100:.4f},"
        f"{np.mean(metrics['f1'])*100:.4f}\n"
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(metrics_line)


def log_gradients(model, epoch, gradients_file):
    """
    Log gradients for each parameter of the model.

    Args:
        model (nn.Module): The model whose gradients should be logged
        epoch (int): Current epoch number
        gradients_file (str): Path to the gradients log file
    """
    with open(gradients_file, "a", encoding="utf-8") as f:
        for name, param in model.named_parameters():
            if param.grad is not None:
                avg_grad = torch.mean(torch.abs(param.grad)).item()
                max_grad = torch.max(torch.abs(param.grad)).item()
                f.write(f"{epoch+1},{name},{avg_grad:.6f},{max_grad:.6f}\n")


def log_learning_rate(lr_file, epoch, optimizer):
    """Log learning rates for each parameter group."""
    with open(lr_file, "a", encoding="utf-8") as f:
        for param_group in optimizer.param_groups:
            f.write(f"{epoch+1},{param_group['lr']:.8f}\n")


def log_resources(resource_file, epoch):
    """Log GPU memory usage and other resource metrics."""
    if torch.cuda.is_available():
        with open(resource_file, "a", encoding="utf-8") as f:
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2  # MB
            f.write(f"{epoch+1},{allocated:.2f},{cached:.2f}\n")


def log_timing(timing_file, epoch, train_time, eval_time, epoch_time):
    """
    Log timing information for training and evaluation.

    Args:
        timing_file (str): Path to the timing log file
        epoch (int): Current epoch number
        train_time (float): Time spent in training
        eval_time (float): Time spent in evaluation
        epoch_time (float): Total time spent in the epoch
    """
    with open(timing_file, "a", encoding="utf-8") as f:
        f.write(f"{epoch+1},{train_time:.2f},{eval_time:.2f},{epoch_time:.2f}\n")


def log_model_info(model_info_file, model, config, hyperparameters):
    """Log model architecture and parameter statistics."""

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info = {
        "model_type": config.model.type,
        "trainable_parameters": count_parameters(model),
        "layer_sizes": {
            name: list(param.size()) for name, param in model.named_parameters()
        },
        "optimizer_type": hyperparameters["optimizer_type"],
        "loss_function": "CrossEntropyLoss",
    }

    with open(model_info_file, "w", encoding="utf-8") as f:
        yaml.dump(model_info, f)
