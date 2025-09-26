import os
import numpy as np
import datetime
from tqdm import tqdm


def save_cv_split_info(cv_split_dir, split_file, dataset_name, cv_outer_fold, train_val_data, test_data, train_val_labels, test_labels, voxel_calculation, seed, suffix=""):
    """
    Save cross-validation split information to a text file for reproducibility and debugging.
    
    Args:
        cv_split_dir (str): Path to cross-validation split directory
        split_file (str): Path to cross-validation split file
        dataset_name (str): Name of the dataset
        cv_outer_fold (int): Cross-validation fold number
        train_val_data (list): Training and validation data paths
        test_data (list): Test data paths
        train_val_labels (list): Training and validation labels
        test_labels (list): Test labels
        voxel_calculation (str): Voxel calculation method used
        seed (int): Random seed used for splitting
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
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Split statistics
        f.write("SPLIT STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {len(train_val_data) + len(test_data)}\n")
        f.write(f"Train+Val Samples: {len(train_val_data)} ({len(train_val_data)/(len(train_val_data) + len(test_data))*100:.1f}%)\n")
        f.write(f"Test Samples: {len(test_data)} ({len(test_data)/(len(train_val_data) + len(test_data))*100:.1f}%)\n\n")
        
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
        for i, (data_path, label) in enumerate(zip(train_val_data, train_val_labels)):
            # Extract sample name from path (e.g., "Lipo-001" from "datasets/lipo_cleaned/preprocessed_median/Lipo-001/image.nii.gz")
            sample_name = os.path.basename(os.path.dirname(data_path))
            f.write(f"{i+1:3d}. {sample_name} (Class {label})\n")
        f.write("\n")
        
        # Test samples
        f.write("TEST SAMPLES:\n")
        f.write("-" * 40 + "\n")
        for i, (data_path, label) in enumerate(zip(test_data, test_labels)):
            # Extract sample name from path
            sample_name = os.path.basename(os.path.dirname(data_path))
            f.write(f"{i+1:3d}. {sample_name} (Class {label})\n")
        f.write("\n")
        
        # Full paths for reference
        f.write("FULL PATHS (for debugging):\n")
        f.write("-" * 40 + "\n")
        f.write("Train+Val Paths:\n")
        for i, data_path in enumerate(train_val_data):
            f.write(f"{i+1:3d}. {data_path}\n")
        
        f.write("\nTest Paths:\n")
        for i, data_path in enumerate(test_data):
            f.write(f"{i+1:3d}. {data_path}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF CV SPLIT INFORMATION\n")
        f.write("=" * 80 + "\n")
    
    print(f"CV split information saved to: {split_file}\n")


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
        'height_min': np.min(heights) if heights else 0,
        'height_max': np.max(heights) if heights else 0,
        'height_mean': np.mean(heights) if heights else 0,
        'depth_min': np.min(depths) if depths else 0,
        'depth_max': np.max(depths) if depths else 0,
        'depth_mean': np.mean(depths) if depths else 0,
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
        f.write(f"Width range: {statistics['width_min']:.1f} - {statistics['width_max']:.1f} (mean: {statistics['width_mean']:.1f})\n")
        f.write(f"Height range: {statistics['height_min']:.1f} - {statistics['height_max']:.1f} (mean: {statistics['height_mean']:.1f})\n")
        f.write(f"Depth range: {statistics['depth_min']:.1f} - {statistics['depth_max']:.1f} (mean: {statistics['depth_mean']:.1f})\n")
        f.write(f"Aspect ratio (width/height): {statistics['aspect_ratio_min']:.2f} - {statistics['aspect_ratio_max']:.2f} (mean: {statistics['aspect_ratio_mean']:.2f})\n")
        f.write(f"Volume range: {statistics['volume_min']:.0f} - {statistics['volume_max']:.0f} (mean: {statistics['volume_mean']:.0f})\n\n")
        
        # Write class distribution
        f.write("=== Class Distribution ===\n")
        total_samples = sum(statistics['class_distribution'].values())
        for label, count in statistics['class_distribution'].items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            f.write(f"Class {label}: {count} samples ({percentage:.1f}%)\n")