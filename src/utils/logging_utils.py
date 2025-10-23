import os

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image, ImageDraw, ImageFont


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
        "metrics": "epoch,phase,loss,accuracy,precision,recall,f1,auc",
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
    log_files, hyperparameters, experimental_setting, model, epochs, pipeline_dir, prev_pipeline_dir
):
    """
    Log model architecture information and print basic run configuration.

    Args:
        log_files (dict): Dictionary containing paths to log files for metrics, gradients,
                         hyperparameters, learning rates, resources, timing, and model info
        hyperparameters (dict): Dictionary of training hyperparameters including optimizer
                               settings, learning rates, and other training parameters
        experimental_setting: Configuration object containing model architecture and training settings
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
    log_model_info(log_files["model_info"], model, experimental_setting, hyperparameters)

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
            - auc (float): AUC value
    """
    metrics_line = (
        f"{epoch+1},"
        f"{phase},"
        f"{metrics['loss']:.4f},"
        f"{np.mean(metrics['accuracy'])*100:.4f},"
        f"{np.mean(metrics['precision'])*100:.4f},"
        f"{np.mean(metrics['recall'])*100:.4f},"
        f"{np.mean(metrics['f1'])*100:.4f},"
        f"{np.mean(metrics['auc'])*100:.4f}\n"
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


def log_model_info(model_info_file, model, experimental_setting, hyperparameters):
    """Log model architecture and parameter statistics."""

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info = {
        "model_type": experimental_setting.model.type,
        "trainable_parameters": count_parameters(model),
        "layer_sizes": {
            name: list(param.size()) for name, param in model.named_parameters()
        },
        "optimizer_type": hyperparameters["optimizer_type"],
        "loss_function": "CrossEntropyLoss",
    }

    with open(model_info_file, "w", encoding="utf-8") as f:
        yaml.dump(model_info, f)


def log_validation_images(writer, model, val_loader, device, fold, epoch):
    """
    Log sample validation images and their predictions to TensorBoard.
    Simplified version for better debugging and performance.

    Args:
        writer: TensorBoard SummaryWriter instance
        model: The neural network model
        val_loader: Validation data loader
        device: Device to run the model on
        fold: Current fold number
        epoch: Current epoch number
    """
    model.eval()
    with torch.no_grad():
        # Get a batch of images - use epoch to get different batches
        batch_idx = epoch % len(val_loader)
        batch = list(val_loader)[batch_idx]
        if isinstance(batch, dict):
            images = batch.get("image")
            labels = batch.get("label")
        else:
            images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Get predictions
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Convert images for TensorBoard (denormalize if necessary)
        if hasattr(val_loader.dataset, "mean") and hasattr(val_loader.dataset, "std"):
            mean = torch.tensor(val_loader.dataset.mean).view(3, 1, 1).to(device)
            std = torch.tensor(val_loader.dataset.std).view(3, 1, 1).to(device)
            images = images * std + mean

        # Process images
        images_with_text = []
        for i in range(min(2, len(images))):
            img = images[i].cpu().numpy()
            
            if len(img.shape) == 4:  # 3D image: [C, H, W, D] or [C, D, H, W]
                # Determine the correct slicing axis based on tensor dimensions
                # For liver dataset: H~400, W~400, D~50
                # The smallest dimension should be the depth (D)
                
                # Find which dimension is the smallest (likely depth)
                dims = img.shape[1:]  # Exclude channel dimension
                depth_dim = np.argmin(dims) + 1  # +1 because we excluded channel
                depth_size = img.shape[depth_dim]
                
                # print(f"DEBUG: Image tensor shape: {img.shape}")
                # print(f"DEBUG: Depth dimension: {depth_dim}, Depth size: {depth_size}")
                
                # Log every 'frequency'th slice for scrollable video in last epoch
                # Starting at slice 'start_slice' and ending at slice 'end_slice'
                # NOTE: If there are too many slices, tensorboard won't be able to display them all.
                # ---------------------------------------------------------------------------------------
                frequency = 1
                start_slice = 0
                end_slice = depth_size
                slice_indices = list(range(start_slice, end_slice, frequency))
                # ---------------------------------------------------------------------------------------
                
                for slice_idx in slice_indices:
                    if depth_dim == 1:  # [C, D, H, W]
                        img_slice = img[:, slice_idx, :, :]  # [C, H, W]
                    elif depth_dim == 2:  # [C, H, D, W]
                        img_slice = img[:, :, slice_idx, :]  # [C, H, W]
                    elif depth_dim == 3:  # [C, H, W, D]
                        img_slice = img[:, :, :, slice_idx]  # [C, H, W]
                    else:
                        raise ValueError(f"Unexpected depth dimension: {depth_dim}")
                    
                    # print(f"DEBUG: Slice {slice_idx} shape: {img_slice.shape}")
                    img_slice = img_slice.transpose(1, 2, 0)  # [H, W, C]
                    
                    # Normalize
                    img_min = np.percentile(img_slice, 5)
                    img_max = np.percentile(img_slice, 95)
                    
                    if img_max > img_min:
                        img_slice = np.clip((img_slice - img_min) / (img_max - img_min), 0, 1)
                    else:
                        img_slice = np.zeros_like(img_slice)

                    # Handle channels
                    if img_slice.shape[-1] == 1:
                        img_slice = np.repeat(img_slice, 3, axis=-1)
                    elif img_slice.shape[-1] != 3:
                        raise ValueError(f"Unexpected number of channels: {img_slice.shape[-1]}")

                    # Convert to PIL and resize with aspect ratio preservation
                    img_slice = (img_slice * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_slice)
                    
                    width, height = img_pil.size
                    # Preserve aspect ratio while ensuring minimum size
                    min_size = 256
                    if width < min_size or height < min_size:
                        # Calculate new dimensions maintaining aspect ratio
                        aspect_ratio = width / height
                        if width < height:
                            new_width = min_size
                            new_height = int(min_size / aspect_ratio)
                        else:
                            new_height = min_size
                            new_width = int(min_size * aspect_ratio)
                        img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
                    
                    final_img = img_pil.copy()
                    draw = ImageDraw.Draw(final_img)

                    # Load font
                    font_size = max(18, final_img.width // 16)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                        except:
                            font = ImageFont.load_default()

                    # Add text
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    
                    text_lines = [
                        f"T:{true_label} P:{pred_label}",
                        f"Slice:{slice_idx}/{depth_size-1} Epoch:{epoch}"
                    ]
                    
                    text_height = len(text_lines) * (font_size + 6) + 16
                    text_width = max(140, final_img.width // 3)
                    
                    draw.rectangle([10, 10, text_width, text_height], fill="white", outline="black", width=3)
                    
                    for j, line in enumerate(text_lines):
                        draw.text((15, 15 + j * (font_size + 6)), line, fill="black", font=font)

                    img_tensor = torch.from_numpy(np.array(final_img).transpose(2, 0, 1)) / 255.0
                    images_with_text.append(img_tensor)
                    
            elif len(img.shape) == 3:  # 2D image: [C, H, W]
                img_slice = img.transpose(1, 2, 0)  # [H, W, C]
                
                # Normalize
                img_min = np.percentile(img_slice, 5)
                img_max = np.percentile(img_slice, 95)
                
                if img_max > img_min:
                    img_slice = np.clip((img_slice - img_min) / (img_max - img_min), 0, 1)
                else:
                    img_slice = np.zeros_like(img_slice)

                if img_slice.shape[-1] == 1:
                    img_slice = np.repeat(img_slice, 3, axis=-1)

                # Convert to PIL and resize with aspect ratio preservation
                img_slice = (img_slice * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_slice)
                
                width, height = img_pil.size
                # Preserve aspect ratio while ensuring minimum size
                min_size = 256
                if width < min_size or height < min_size:
                    # Calculate new dimensions maintaining aspect ratio
                    aspect_ratio = width / height
                    if width < height:
                        new_width = min_size
                        new_height = int(min_size / aspect_ratio)
                    else:
                        new_height = min_size
                        new_width = int(min_size * aspect_ratio)
                    img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
                
                final_img = img_pil.copy()
                draw = ImageDraw.Draw(final_img)

                # Load font
                font_size = max(18, final_img.width // 16)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                    except:
                        font = ImageFont.load_default()

                # Add text
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                text_lines = [
                    f"T:{true_label} P:{pred_label}",
                    f"2D E:{epoch}"
                ]
                
                text_height = len(text_lines) * (font_size + 6) + 16
                text_width = max(140, final_img.width // 3)
                
                draw.rectangle([10, 10, text_width, text_height], fill="white", outline="black", width=3)
                
                for j, line in enumerate(text_lines):
                    draw.text((15, 15 + j * (font_size + 6)), line, fill="black", font=font)

                img_tensor = torch.from_numpy(np.array(final_img).transpose(2, 0, 1)) / 255.0
                images_with_text.append(img_tensor)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

        # Log each slice as separate epoch for scrollable video
        if images_with_text:
            # Log each slice as a separate "epoch" for video-like scrolling
            for slice_idx, img_tensor in enumerate(images_with_text):
                writer.add_image(f"Images/fold_{fold}", img_tensor, slice_idx)
