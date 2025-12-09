import os
import csv
import datetime
import json
from pathlib import Path

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
        # IMPORTANT: Don't convert DataLoader to list (causes "too many open files" error)
        # Instead, iterate directly to the desired batch index
        batch_idx = epoch % len(val_loader)
        batch = None
        for i, b in enumerate(val_loader):
            if i == batch_idx:
                batch = b
                break
            # Clean up intermediate batches to avoid memory/file descriptor accumulation
            del b
        
        if batch is None:
            # Fallback: if batch_idx is out of range, use first batch
            batch = next(iter(val_loader))
        
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


def save_cv_summary(experimental_setting, cv_outer_folds):
    """
    Save cross-validation summary to a text file.
    
    Args:
        experimental_setting (DictConfig): Hydra configuration object
        cv_outer_folds (int): Number of cross-validation folds
    
    Returns:
        str: Path to the created summary file
    """
    # Create summary directory
    summary_dir = os.path.join(experimental_setting.experiment_base_dir, "cv_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create summary file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(summary_dir, f"cv_summary_{timestamp}.txt")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Experiment information
        f.write("EXPERIMENT INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {experimental_setting.data.dataset}\n")
        f.write(f"Dimensionality: {experimental_setting.data.dimensionality}\n")
        f.write(f"Voxel Calculation: {experimental_setting.data.voxel_calculation}\n")
        f.write(f"Number of Outer Cross-Validation Folds: {cv_outer_folds}\n")
        f.write(f"N Repeats: {experimental_setting.cv_outer_folds_repeats}\n")
        f.write(f"N Splits per Repeat: {experimental_setting.cv_outer_folds_splits}\n")
        f.write(f"Seed: {experimental_setting.seed}\n")
        f.write(f"Max Evaluations: {experimental_setting.max_evaluations}\n")
        # Support both NePS and QuickTune
        if hasattr(experimental_setting, "searcher"):
            f.write(f"Optimizer: {experimental_setting.searcher}\n")
        elif hasattr(experimental_setting, "qt"):
            f.write(f"Optimizer: QuickTune\n")
        f.write(f"Developer Mode: {experimental_setting.developer_mode}\n")
        f.write(f"Number of Epochs: {experimental_setting.training.number_of_epochs}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # CV Fold directories
        f.write("CROSS-VALIDATION FOLD DIRECTORIES:\n")
        f.write("-" * 40 + "\n")
        # Support both NePS and QuickTune directory structures
        if hasattr(experimental_setting, "neps_directory") and experimental_setting.neps_directory:
            base_cv_dir = experimental_setting.neps_directory
        else:
            # For QuickTune, CV folds are directly under experiment_base_dir
            base_cv_dir = experimental_setting.experiment_base_dir
        for cv_outer_fold in range(cv_outer_folds):
            cv_dir = f"{base_cv_dir}/cv_outer_fold_{cv_outer_fold}"
            f.write(f"CV Fold {cv_outer_fold}: {cv_dir}\n")
        f.write("\n")
        
        # Configuration files
        f.write("CONFIGURATION FILES:\n")
        f.write("-" * 40 + "\n")
        config_dir = os.path.join(experimental_setting.experiment_base_dir, "hydra_output")
        f.write(f"Configuration Directory: {config_dir}\n")
        f.write("Files:\n")
        f.write("  - experimental_setting.yaml\n")
        f.write("  - pipeline_space.yaml\n")
        f.write("  - pipeline_space_compact.yaml\n\n")
        
        # Data information
        f.write("DATA INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Data Path: {experimental_setting.data.path}\n")
        f.write(f"Cache Data: {experimental_setting.data.cache_data}\n")
        f.write(f"Use Smart Preprocessing: {experimental_setting.data.use_smart_preprocessing}\n")
        cv_inner_folds_splits = experimental_setting.cv_inner_folds_splits if hasattr(experimental_setting, "cv_inner_folds_splits") else 5
        cv_inner_folds_repeats = experimental_setting.cv_inner_folds_repeats if hasattr(experimental_setting, "cv_inner_folds_repeats") else 1
        total_inner_folds = cv_inner_folds_repeats * cv_inner_folds_splits
        f.write(f"K-Folds: {total_inner_folds} (repeats: {cv_inner_folds_repeats}, splits: {cv_inner_folds_splits})\n")
        f.write(f"Num Workers: {experimental_setting.data.num_workers}\n\n")
        
        # Pipeline space information
        f.write("PIPELINE SPACE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pipeline Space File: {experimental_setting.pipeline_space}\n")
        f.write(f"Developer Mode Pipeline: {experimental_setting.developer_mode}\n\n")
        
        # Summary
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total NePS Runs: {cv_outer_folds}\n")
        f.write(f"Each run uses different train+val/test split\n")
        f.write(f"Results saved in separate directories per fold\n")
        f.write(f"Cross-validation ensures robust evaluation\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nCross-validation summary saved to: {summary_file}")
    return summary_file


def update_performances_csv_from_neps_output(neps_output_dir: str, cv_outer_fold: int) -> None:
    """
    Update incumbent_performances.csv file with validation and test performances from all configs.
    
    This function scans all config directories in the current outer fold, reads their
    report.yaml and test_evaluation_results.json files, calculates incumbent performances,
    and creates/updates an incumbent_performances.csv file.
    
    Args:
        neps_output_dir: Path to the NePS output directory for one outer fold (e.g., .../NePS_output/cv_outer_fold_0)
        cv_outer_fold: Current outer fold number
    """
    neps_output_path = Path(neps_output_dir)
    if not neps_output_path.exists():
        return
    
    # Path to incumbent_performances.csv in main NePS output directory (parent of outer fold)
    main_neps_output = neps_output_path.parent if "cv_outer_fold_" in str(neps_output_path) else neps_output_path
    performances_csv_path = main_neps_output / "incumbent_performances.csv"
    
    # Collect all performance data from all outer folds
    all_fold_data = {}  # outer_fold -> {config_num -> {"validation": val, "test": test}}
    
    # Find all outer fold directories
    outer_fold_dirs = sorted(
        [d for d in main_neps_output.iterdir() 
         if d.is_dir() and d.name.startswith("cv_outer_fold_")],
        key=lambda x: int(x.name.split("_")[-1])
    )
    
    for outer_fold_dir in outer_fold_dirs:
        fold_num = int(outer_fold_dir.name.split("_")[-1])
        configs_dir = outer_fold_dir / "configs"
        
        if not configs_dir.exists():
            continue
        
        config_dirs = sorted(
            [d for d in configs_dir.iterdir() if d.is_dir() and d.name.startswith("config_")],
            key=lambda x: int(x.name.split("_")[-1])
        )
        
        all_fold_data[fold_num] = {}
        
        for config_dir in config_dirs:
            config_num = int(config_dir.name.split("_")[-1])
            
            # Read validation performance from report.yaml
            report_path = config_dir / "report.yaml"
            validation_perf = None
            if report_path.exists():
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report = yaml.safe_load(f)
                    objective = report.get("objective_to_minimize", None)
                    if objective is not None:
                        validation_perf = abs(objective)  # Remove negative sign
                except Exception:
                    pass
            
            # Read test performance from test_evaluation_results.json
            test_results_path = config_dir / "test_evaluation_results.json"
            test_perf = None
            if test_results_path.exists():
                try:
                    with open(test_results_path, "r", encoding="utf-8") as f:
                        test_results = json.load(f)
                    ensemble = test_results.get("ensemble", {})
                    test_perf = ensemble.get("auc_macro", None)
                except Exception:
                    pass
            
            if validation_perf is not None or test_perf is not None:
                all_fold_data[fold_num][config_num] = {
                    "validation": validation_perf,
                    "test": test_perf
                }
    
    # Calculate incumbent performances for each fold
    # Structure: fold -> config -> {"validation_incumbent": val, "test": test}
    incumbent_data = {}
    
    for fold_num in sorted(all_fold_data.keys()):
        fold_configs = all_fold_data[fold_num]
        if not fold_configs:
            continue
        
        incumbent_data[fold_num] = {}
        best_val_so_far = float('-inf')
        best_val_config = None
        
        for config_num in sorted(fold_configs.keys()):
            config_data = fold_configs[config_num]
            val_perf = config_data.get("validation")
            test_perf = config_data.get("test")
            
            # Update validation incumbent
            if val_perf is not None:
                if val_perf > best_val_so_far:
                    best_val_so_far = val_perf
                    best_val_config = config_num
            
            # Store incumbent validation and test (of best validation config)
            # Note: best_val_so_far should never be -inf at this point if we have configs
            incumbent_data[fold_num][config_num] = {
                "validation_incumbent": best_val_so_far if best_val_so_far != float('-inf') else None,
                "test": fold_configs[best_val_config].get("test") if best_val_config is not None and best_val_config in fold_configs else test_perf
            }
    
    # Write CSV file
    if not incumbent_data:
        return
    
    # Get all config numbers across all folds and find maximum
    all_configs = set()
    for fold_data in incumbent_data.values():
        all_configs.update(fold_data.keys())
    max_config = max(all_configs) if all_configs else 0
    all_configs = sorted(range(1, max_config + 1))  # Include all configs from 1 to max_config
    
    # Fill missing configs with last incumbent value for each fold
    for fold_num in sorted(incumbent_data.keys()):
        fold_incumbents = incumbent_data[fold_num]
        if not fold_incumbents:
            continue
        
        # Find the last config that exists in this fold and get its incumbent values
        # We need to find the last config that has valid (non-None) values
        last_val_incumbent = None
        last_test = None
        
        # Iterate through configs in reverse order to find the last valid values
        for config_num in sorted(fold_incumbents.keys(), reverse=True):
            config_data = fold_incumbents[config_num]
            if last_val_incumbent is None and config_data["validation_incumbent"] is not None:
                last_val_incumbent = config_data["validation_incumbent"]
            if last_test is None and config_data["test"] is not None:
                last_test = config_data["test"]
            # If we found both, we can break early
            if last_val_incumbent is not None and last_test is not None:
                break
        
        # Fill missing configs with last incumbent values
        for config_num in all_configs:
            if config_num not in fold_incumbents:
                # Use last incumbent values for missing configs
                # Only fill if we have valid values
                if last_val_incumbent is not None or last_test is not None:
                    incumbent_data[fold_num][config_num] = {
                        "validation_incumbent": last_val_incumbent,
                        "test": last_test
                    }
    
    # Get all fold numbers
    all_folds = sorted(incumbent_data.keys())
    max_folds = len(all_folds)
    
    # Create CSV rows
    csv_rows = []
    
    # Header
    header = ["config"]
    for fold_idx in range(max_folds):
        header.append(f"validation_fold_{fold_idx}")
    for fold_idx in range(max_folds):
        header.append(f"test_fold_{fold_idx}")
    header.extend(["validation_mean", "validation_std", "test_mean", "test_std"])
    csv_rows.append(header)
    
    # Data rows
    for config_num in all_configs:
        row = [config_num]
        
        # Validation incumbent performances per fold
        val_perfs = []
        for fold_num in all_folds:
            if config_num in incumbent_data[fold_num]:
                val_inc = incumbent_data[fold_num][config_num]["validation_incumbent"]
                row.append(val_inc if val_inc is not None else "")
                if val_inc is not None:
                    val_perfs.append(val_inc)
            else:
                row.append("")
        
        # Test performances per fold
        test_perfs = []
        for fold_num in all_folds:
            if config_num in incumbent_data[fold_num]:
                test_val = incumbent_data[fold_num][config_num]["test"]
                row.append(test_val if test_val is not None else "")
                if test_val is not None:
                    test_perfs.append(test_val)
            else:
                row.append("")
        
        # Calculate mean and std
        val_mean = np.mean(val_perfs) if val_perfs else ""
        val_std = np.std(val_perfs) if val_perfs and len(val_perfs) > 1 else (0.0 if val_perfs else "")
        test_mean = np.mean(test_perfs) if test_perfs else ""
        test_std = np.std(test_perfs) if test_perfs and len(test_perfs) > 1 else (0.0 if test_perfs else "")
        
        row.extend([val_mean, val_std, test_mean, test_std])
        csv_rows.append(row)
    
    # Write CSV file
    with open(performances_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"Updated incumbent_performances.csv: {performances_csv_path}")


def update_cost_csv_from_neps_output(neps_output_dir: str) -> None:
    """
    Update cost CSV files with cost and evaluation_duration from all report.yaml files.
    
    This function scans all outer fold directories in the NePS output directory, reads all
    report.yaml files from all configs, and creates/updates three CSV files with cost
    and evaluation_duration data across all outer folds:
    - costs_in_sec.csv: values in seconds
    - costs_in_min.csv: values in minutes
    - costs_in_hours.csv: values in hours
    
    Args:
        neps_output_dir: Path to the main NePS output directory (e.g., .../NePS_output)
    """
    neps_output_path = Path(neps_output_dir)
    if not neps_output_path.exists():
        print(f"Warning: NePS output directory not found: {neps_output_path}")
        return
    
    # Find all outer fold directories
    outer_fold_dirs = sorted(
        [d for d in neps_output_path.iterdir() 
         if d.is_dir() and d.name.startswith("cv_outer_fold_")],
        key=lambda x: int(x.name.split("_")[-1])
    )
    
    if not outer_fold_dirs:
        print(f"Warning: No outer fold directories found in {neps_output_path}")
        return
    
    # Collect all cost data from all outer folds
    cost_data = []
    for outer_fold_dir in outer_fold_dirs:
        # Extract outer fold number
        cv_outer_fold = int(outer_fold_dir.name.split("_")[-1])
        
        # Find all config directories in this outer fold
        configs_dir = outer_fold_dir / "configs"
        if not configs_dir.exists():
            continue
        
        config_dirs = sorted(
            [d for d in configs_dir.iterdir() if d.is_dir() and d.name.startswith("config_")],
            key=lambda x: int(x.name.split("_")[-1])
        )
        
        for config_dir in config_dirs:
            config_number = int(config_dir.name.split("_")[-1])
            report_path = config_dir / "report.yaml"
            
            if not report_path.exists():
                continue
            
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = yaml.safe_load(f)
                
                cost = report.get("cost", None)
                evaluation_duration = report.get("evaluation_duration", None)
                
                if cost is None or evaluation_duration is None:
                    continue
                
                cost_data.append({
                    "outer_fold": cv_outer_fold,
                    "config": config_number,
                    "cost": cost,
                    "evaluation_duration": evaluation_duration
                })
                
            except Exception as e:
                print(f"Warning: Could not read {report_path}: {e}")
                continue
    
    # Write/overwrite CSV files with all collected data (in seconds, minutes, and hours)
    if cost_data:
        # Sort by outer_fold first, then by config number
        cost_data.sort(key=lambda x: (x["outer_fold"], x["config"]))
        
        # Define CSV file paths
        csv_files = {
            "costs_in_sec.csv": 1.0,      # No conversion (already in seconds)
            "costs_in_min.csv": 1.0 / 60.0,  # Convert to minutes
            "costs_in_hours.csv": 1.0 / 3600.0  # Convert to hours
        }
        
        for csv_filename, conversion_factor in csv_files.items():
            csv_path = neps_output_path / csv_filename
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(["outer_fold", "config", "cost", "evaluation_duration"])
                
                # Write data rows with converted values
                for row in cost_data:
                    cost_converted = row["cost"] * conversion_factor
                    duration_converted = row["evaluation_duration"] * conversion_factor
                    writer.writerow([row["outer_fold"], row["config"], cost_converted, duration_converted])
            
            print(f"Updated {csv_filename} with {len(cost_data)} config(s) across {len(outer_fold_dirs)} outer fold(s): {csv_path}")
    else:
        print(f"Warning: No cost data found to write to CSV files")
