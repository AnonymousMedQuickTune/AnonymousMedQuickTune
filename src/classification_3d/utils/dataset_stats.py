import os
import numpy as np

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
    
    for img_dir in image_dirs:
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