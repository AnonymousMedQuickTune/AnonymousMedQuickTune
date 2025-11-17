import os
import numpy as np
import datetime
import re
from pathlib import Path
from tqdm import tqdm


def save_cv_split_info(cv_split_dir, split_file, dataset_name, cv_outer_fold, train_val_images, test_images, train_val_labels, test_labels, voxel_calculation, seed, suffix="", is_medmnist=False):
    """
    Save cross-validation split information to a text file for reproducibility and debugging.
    
    Args:
        cv_split_dir (str): Path to cross-validation split directory
        split_file (str): Path to cross-validation split file
        dataset_name (str): Name of the dataset
        cv_outer_fold (int): Cross-validation fold number
        train_val_images (list): Training and validation images (paths for WORC, numpy arrays for MedMNIST)
        test_images (list): Test images (paths for WORC, numpy arrays for MedMNIST)
        train_val_labels (list): Training and validation labels
        test_labels (list): Test labels
        voxel_calculation (str): Voxel calculation method used
        seed (int): Random seed used for splitting
        suffix (str): Optional suffix for the filename
        is_medmnist (bool): If True, images are numpy arrays instead of file paths
    """
    os.makedirs(cv_split_dir, exist_ok=True)
    
    with open(split_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CROSS-VALIDATION SPLIT INFORMATION - CV FOLD {cv_outer_fold}\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset information
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"CV Fold: {cv_outer_fold}\n")
        f.write(f"Voxel Calculation: {voxel_calculation}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"CV Seed (seed + cv_outer_fold): {seed + cv_outer_fold}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if is_medmnist:
            f.write(f"Data Format: MedMNIST (numpy arrays)\n")
        else:
            f.write(f"Data Format: WORC (file paths)\n")
        f.write("\n")
        
        # Split statistics
        f.write("SPLIT STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {len(train_val_images) + len(test_images)}\n")
        f.write(f"Train+Val Samples: {len(train_val_images)} ({len(train_val_images)/(len(train_val_images) + len(test_images))*100:.1f}%)\n")
        f.write(f"Test Samples: {len(test_images)} ({len(test_images)/(len(train_val_images) + len(test_images))*100:.1f}%)\n\n")
        
        # Class distribution
        f.write("CLASS DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        train_val_unique, train_val_counts = np.unique(train_val_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        
        f.write("Train+Val Set:\n")
        for label, count in zip(train_val_unique, train_val_counts):
            f.write(f"  Class {label}: {count} samples\n")
        
        f.write("\nTest Set:\n")
        for label, count in zip(test_unique, test_counts):
            f.write(f"  Class {label}: {count} samples\n")
        f.write("\n")
        
        # Train+Val samples
        f.write("TRAIN+VAL SAMPLES:\n")
        f.write("-" * 40 + "\n")
        for i, (data_item, label) in enumerate(zip(train_val_images, train_val_labels)):
            if is_medmnist:
                # For MedMNIST: data_item is a numpy array, use index as identifier
                if isinstance(data_item, np.ndarray):
                    shape_str = f"shape={data_item.shape}"
                else:
                    shape_str = "numpy_array"
                f.write(f"{i+1:3d}. Sample {i+1} (Class {label}, {shape_str})\n")
            else:
                # For WORC: data_item is a file path
                sample_name = os.path.basename(os.path.dirname(data_item))
                f.write(f"{i+1:3d}. {sample_name} (Class {label})\n")
        f.write("\n")
        
        # Test samples
        f.write("TEST SAMPLES:\n")
        f.write("-" * 40 + "\n")
        for i, (data_item, label) in enumerate(zip(test_images, test_labels)):
            if is_medmnist:
                # For MedMNIST: data_item is a numpy array, use index as identifier
                if isinstance(data_item, np.ndarray):
                    shape_str = f"shape={data_item.shape}"
                else:
                    shape_str = "numpy_array"
                f.write(f"{i+1:3d}. Sample {i+1} (Class {label}, {shape_str})\n")
            else:
                # For WORC: data_item is a file path
                sample_name = os.path.basename(os.path.dirname(data_item))
                f.write(f"{i+1:3d}. {sample_name} (Class {label})\n")
        f.write("\n")
        
        # Full paths/references for debugging
        if is_medmnist:
            f.write("SAMPLE INFORMATION (for debugging):\n")
            f.write("-" * 40 + "\n")
            f.write("Train+Val Samples (first 10):\n")
            for i, (data_item, label) in enumerate(zip(train_val_images[:10], train_val_labels[:10])):
                if isinstance(data_item, np.ndarray):
                    f.write(f"{i+1:3d}. Shape: {data_item.shape}, Class: {label}, Dtype: {data_item.dtype}, Range: [{data_item.min():.2f}, {data_item.max():.2f}]\n")
                else:
                    f.write(f"{i+1:3d}. {type(data_item).__name__}, Class: {label}\n")
            if len(train_val_images) > 10:
                f.write(f"... ({len(train_val_images) - 10} more samples)\n")
            
            f.write("\nTest Samples (first 10):\n")
            for i, (data_item, label) in enumerate(zip(test_images[:10], test_labels[:10])):
                if isinstance(data_item, np.ndarray):
                    f.write(f"{i+1:3d}. Shape: {data_item.shape}, Class: {label}, Dtype: {data_item.dtype}, Range: [{data_item.min():.2f}, {data_item.max():.2f}]\n")
                else:
                    f.write(f"{i+1:3d}. {type(data_item).__name__}, Class: {label}\n")
            if len(test_images) > 10:
                f.write(f"... ({len(test_images) - 10} more samples)\n")
        else:
            f.write("FULL PATHS (for debugging):\n")
            f.write("-" * 40 + "\n")
            f.write("Train+Val Paths:\n")
            for i, data_path in enumerate(train_val_images):
                f.write(f"{i+1:3d}. {data_path}\n")
            
            f.write("\nTest Paths:\n")
            for i, data_path in enumerate(test_images):
                f.write(f"{i+1:3d}. {data_path}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF CV SPLIT INFORMATION\n")
        f.write("=" * 80 + "\n")
    
    print(f"CV split information saved to: {split_file}")


def analyze_dataset_statistics(cleaned_path, labels_df):
    """
    Analyze 3D medical image dataset statistics.
    
    Args:
        cleaned_path (str): Path to cleaned dataset
        labels_df (pd.DataFrame): DataFrame with labels
        
    Returns:
        dict: Dictionary containing statistics
    """
    import nibabel as nib
    import numpy as np
    from collections import Counter
    
    print("Analyzing image properties...")
    
    # Get all image directories
    image_dirs = [d for d in os.listdir(cleaned_path) 
                  if os.path.isdir(os.path.join(cleaned_path, d))]
    
    # Analyze image properties
    widths = []
    heights = []
    depths = []
    volumes = []
    aspect_ratios = []
    
    print(f"Analyzing {len(image_dirs)} images...")
    for img_dir in tqdm(image_dirs, desc="", unit="img"):
        img_path = os.path.join(cleaned_path, img_dir, "image.nii.gz")
        if os.path.exists(img_path):
            try:
                img = nib.load(img_path)
                img_data = img.get_fdata()
                
                # Get dimensions - NIfTI files have shape (Width, Height, Depth)
                width, height, depth = img_data.shape
                widths.append(width)
                heights.append(height)
                depths.append(depth)
                volumes.append(width * height * depth)
                aspect_ratios.append(width / height)
                
            except Exception as e:
                print(f"Warning: Could not analyze {img_path}: {e}")
                continue
    
    # Calculate statistics
    statistics = {
        'width_min': np.min(widths) if widths else 0,
        'width_max': np.max(widths) if widths else 0,
        'width_mean': np.mean(widths) if widths else 0,
        'width_p95': np.percentile(widths, 95) if widths else 0,
        'width_p99': np.percentile(widths, 99) if widths else 0,
        'height_min': np.min(heights) if heights else 0,
        'height_max': np.max(heights) if heights else 0,
        'height_mean': np.mean(heights) if heights else 0,
        'height_p95': np.percentile(heights, 95) if heights else 0,
        'height_p99': np.percentile(heights, 99) if heights else 0,
        'depth_min': np.min(depths) if depths else 0,
        'depth_max': np.max(depths) if depths else 0,
        'depth_mean': np.mean(depths) if depths else 0,
        'depth_p95': np.percentile(depths, 95) if depths else 0,
        'depth_p99': np.percentile(depths, 99) if depths else 0,
        'volume_min': np.min(volumes) if volumes else 0,
        'volume_max': np.max(volumes) if volumes else 0,
        'volume_mean': np.mean(volumes) if volumes else 0,
        'aspect_ratio_min': np.min(aspect_ratios) if aspect_ratios else 0,
        'aspect_ratio_max': np.max(aspect_ratios) if aspect_ratios else 0,
        'aspect_ratio_mean': np.mean(aspect_ratios) if aspect_ratios else 0,
    }
    
    # Analyze class distribution
    if 'Diagnosis_binary' in labels_df.columns:
        labels = labels_df['Diagnosis_binary'].values
        class_counts = Counter(labels)
        statistics['class_distribution'] = dict(class_counts)
    else:
        statistics['class_distribution'] = {}
    
    # Print summary
    print(f"Analyzed {len(widths)} images")
    print(f"Dimensions: {statistics['width_min']:.0f}-{statistics['width_max']:.0f} x {statistics['height_min']:.0f}-{statistics['height_max']:.0f} x {statistics['depth_min']:.0f}-{statistics['depth_max']:.0f}")
    print(f"Volumes: {statistics['volume_min']:.0f}-{statistics['volume_max']:.0f} voxels")
    print(f"Class distribution: {statistics['class_distribution']}")
    
    return statistics


def save_statistics_to_file(statistics, output_file, dataset_name, additional_info=None):
    """
    Save dataset statistics to a file in a standardized format.
    
    Args:
        statistics (dict): Statistics dictionary from analyze_dataset_statistics
        output_file (str): Path to output file
        dataset_name (str): Name of the dataset
        additional_info (dict, optional): Additional information to include in the header
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write("=== Dataset Statistics ===\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        
        # Write additional info if provided
        if additional_info:
            for key, value in additional_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Write image properties
        f.write("=== Image Properties ===\n")
        f.write(f"Width range: {statistics['width_min']:.1f} - {statistics['width_max']:.1f} (mean: {statistics['width_mean']:.1f}, p95: {statistics.get('width_p95', 0):.1f}, p99: {statistics.get('width_p99', 0):.1f})\n")
        f.write(f"Height range: {statistics['height_min']:.1f} - {statistics['height_max']:.1f} (mean: {statistics['height_mean']:.1f}, p95: {statistics.get('height_p95', 0):.1f}, p99: {statistics.get('height_p99', 0):.1f})\n")
        f.write(f"Depth range: {statistics['depth_min']:.1f} - {statistics['depth_max']:.1f} (mean: {statistics['depth_mean']:.1f}, p95: {statistics.get('depth_p95', 0):.1f}, p99: {statistics.get('depth_p99', 0):.1f})\n")
        f.write(f"Aspect ratio (width/height): {statistics['aspect_ratio_min']:.2f} - {statistics['aspect_ratio_max']:.2f} (mean: {statistics['aspect_ratio_mean']:.2f})\n")
        f.write(f"Volume range: {statistics['volume_min']:.0f} - {statistics['volume_max']:.0f} (mean: {statistics['volume_mean']:.0f})\n\n")
        
        # Write class distribution
        f.write("=== Class Distribution ===\n")
        total_samples = sum(statistics['class_distribution'].values())
        for label, count in statistics['class_distribution'].items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            f.write(f"Class {label}: {count} samples ({percentage:.1f}%)\n")


def _read_dimensions_from_statistics(statistics_file, use_percentile=True, percentile=95):
    """
    Read dimensions from statistics.txt file.
    Can use either maximum (conservative) or percentile-based (efficient) approach.
    
    Args:
        statistics_file (str): Path to statistics.txt file
        use_percentile (bool): If True, use percentile instead of maximum (default: True)
        percentile (int): Percentile to use if use_percentile=True (default: 95 for 95th percentile)
        
    Returns:
        tuple: (height, width, depth) as integers
    """
    if not os.path.exists(statistics_file):
        raise FileNotFoundError(f"Statistics file not found: {statistics_file}")
    
    with open(statistics_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    if use_percentile:
        # Try to parse percentile values (new format)
        # Format: "Height range: 192.0 - 560.0 (mean: 430.3, p95: 520.0, p99: 550.0)"
        percentile_key = f"p{percentile}"
        
        # More flexible regex that handles various whitespace patterns
        height_match = re.search(r"Height range:[\d.\s-]+\(mean:\s*[\d.]+,\s*" + percentile_key + r":\s*([\d.]+)", content)
        width_match = re.search(r"Width range:[\d.\s-]+\(mean:\s*[\d.]+,\s*" + percentile_key + r":\s*([\d.]+)", content)
        depth_match = re.search(r"Depth range:[\d.\s-]+\(mean:\s*[\d.]+,\s*" + percentile_key + r":\s*([\d.]+)", content)
        
        if height_match and width_match and depth_match:
            height = int(float(height_match.group(1)))
            width = int(float(width_match.group(1)))
            depth = int(float(depth_match.group(1)))
            print(f"Using {percentile}th percentile from statistics: Height={height}, Width={width}, Depth={depth}")
            return height, width, depth
        else:
            # Fallback: old format without percentiles, use maximum
            print(f"Warning: Percentile {percentile} not found in {statistics_file}, falling back to maximum")
            use_percentile = False
    
    # Fallback to maximum (old format or percentile not found)
    # Parse height range: "Height range: 192.0 - 560.0 (mean: 430.3)"
    height_match = re.search(r"Height range:\s*[\d.]+ - ([\d.]+)", content)
    width_match = re.search(r"Width range:\s*[\d.]+ - ([\d.]+)", content)
    depth_match = re.search(r"Depth range:\s*[\d.]+ - ([\d.]+)", content)
    
    if not height_match or not width_match or not depth_match:
        raise ValueError(f"Could not parse dimensions from {statistics_file}")
    
    height = int(float(height_match.group(1)))
    width = int(float(width_match.group(1)))
    depth = int(float(depth_match.group(1)))
    
    print(f"Using maximum dimensions (conservative): Height={height}, Width={width}, Depth={depth}")
    return height, width, depth


def extract_spatial_size(model_type, voxel_calculation, dataset_name, developer_mode, data_path, use_percentile=True, percentile=95):
    """
    Extract spatial size based on model type and voxel calculation method.
    Uses intelligent percentile-based approach instead of maximum to improve efficiency.
    
    Instead of using the maximum dimensions (which requires padding all smaller images),
    this function uses a percentile-based approach (default: 95th percentile) that:
    - Covers ~95% of images without cropping
    - Only requires cropping for the largest ~5% of images
    - Significantly reduces memory usage and computation time
    
    Args:
        model_type (str): Type of model (e.g., "vit", "densenet", etc.)
        voxel_calculation (str): Voxel calculation method ("mean", "median", "isotropic", "volumetric_isotropic")
        dataset_name (str): Name of the dataset
        developer_mode (bool): Whether the developer mode is enabled
        data_path (str): Path to the datasets directory
        use_percentile (bool): If True, use percentile-based approach (default: True, recommended)
        percentile (int): Percentile to use (default: 95 for 95th percentile)
        
    Returns:
        tuple: Spatial size in (H, W, D) format, or None if not needed
    """
    if developer_mode:
        spatial_size = (64, 64, 32)  # spatial size in (H, W, D) format
    else:
        # Only ViT and SwinUNETR need specific spatial sizes
        if model_type in ["vit", "swin_unetr"]:
            # Construct path to statistics.txt file
            statistics_file = os.path.join(
                data_path,
                f"{dataset_name}_cleaned",
                f"preprocessed_{voxel_calculation}",
                "statistics.txt"
            )
            
            # Read dimensions from statistics.txt (using percentile or maximum)
            try:
                height, width, depth = _read_dimensions_from_statistics(
                    statistics_file,
                    use_percentile=use_percentile,
                    percentile=percentile
                )
                spatial_size = (height, width, depth)  # (H, W, D) format
                print(f"Extracted spatial size from {statistics_file}: {spatial_size}")
            except Exception as e:
                print(f"Warning: Could not read spatial size from {statistics_file}: {e}")
                raise NotImplementedError(
                    f"Could not extract spatial size for {dataset_name} with {voxel_calculation} voxel calculation. "
                    f"Please ensure the statistics.txt file exists at: {statistics_file}"
                )
            
            # Update spatial size to be divisible by 32 (round down)
            h = (spatial_size[0] // 32) * 32
            w = (spatial_size[1] // 32) * 32
            d = (spatial_size[2] // 32) * 32
            spatial_size = (h, w, d)
            print(f"Spatial size after rounding to multiples of 32: {spatial_size}")

        else:
            spatial_size = None
    
    return spatial_size