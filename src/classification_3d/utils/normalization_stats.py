import numpy as np
import torch
from monai.transforms import LoadImage

def autonorm(hyperparameters, is_rgb=False):  
    """
    Get normalization parameters from NePS hyperparameters.
    Values are normalized to be between 0 and 1.
    
    Args:
        hyperparameters (dict): NePS hyperparameters
        is_rgb (bool): Whether the dataset is RGB or grayscale. For RGB datasets we have 3 channels, for grayscale we have 1 channel.
    
    Returns:
        dict: Dictionary containing normalization parameters with keys 'mean' and 'std' (list format for compatibility)
    """
    # Use normalization stats from NePS hyperparameters    
    if is_rgb:
        mean_values = [
            float(hyperparameters.get("mean_1", 0.0)),
            float(hyperparameters.get("mean_2", 0.0)),
            float(hyperparameters.get("mean_3", 0.0)),
        ]
        std_values = [
            float(hyperparameters.get("std_1", 1.0)),
            float(hyperparameters.get("std_2", 1.0)),
            float(hyperparameters.get("std_3", 1.0)),
        ]
    else:
        mean_values = [float(hyperparameters.get("mean", 0.0))]
        std_values = [float(hyperparameters.get("std", 1.0))]

    normalization_stats = {"mean": mean_values, "std": std_values}

    return normalization_stats


def calculate_normalization_stats(train_data, is_rgb=False): 
    """
    Calculate normalization statistics for 3D medical CT images.
    
    Collect all intensity values from training data, clip to [0.5, 99.5] percentiles,
    then calculate mean and std for Z-score normalization.
    
    Args:
        train_data (list): List of dictionaries containing 'image' key with either:
            - file paths to 3D volumes (str) for WORC datasets
            - numpy arrays (np.ndarray) for MedMNIST datasets
        is_rgb (bool): Whether the dataset is RGB or grayscale (not supported yet)
    
    Returns:
        dict: Dictionary containing normalization parameters with keys 'mean' and 'std'
    """
    
    print("Calculating CT normalization statistics from preprocessed training data...")
    
    # Collect all intensity values from training images
    all_intensities = []
    
    # Load and process each image
    loader = LoadImage(image_only=True)
    
    for data_dict in train_data:
        img_data = data_dict['image']
        
        # Check if image_data is a numpy array (MedMNIST) or a file path (WORC)
        if isinstance(img_data, np.ndarray):
            # MedMNIST: image is already a numpy array
            img = img_data
        elif isinstance(img_data, str):
            # WORC: Load the image from file path
            img = loader(img_data)
        else:
            raise TypeError(f"Unsupported image type: {type(img_data)}. Expected str (file path) or np.ndarray.")
        
        # Convert to numpy array if it's a torch tensor
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        # Flatten the image to get all intensity values
        img_flat = img.flatten()
        
        # Add to collection
        all_intensities.extend(img_flat)
    
    # Convert to numpy array for efficient computation
    all_intensities = np.array(all_intensities)
    
    # Clip to [0.5, 99.5] percentiles
    lower_percentile = np.percentile(all_intensities, 0.5)
    upper_percentile = np.percentile(all_intensities, 99.5)
    
    print(f"CT intensity range: [{np.min(all_intensities):.2f}, {np.max(all_intensities):.2f}]")
    print(f"CT percentiles [0.5, 99.5]: [{lower_percentile:.2f}, {upper_percentile:.2f}]")
    
    # Clip intensities to percentiles
    clipped_intensities = np.clip(all_intensities, lower_percentile, upper_percentile)
    
    # Calculate mean and std for Z-score normalization
    mean_value = float(np.mean(clipped_intensities))
    std_value = float(np.std(clipped_intensities))
    
    print(f"CT normalization stats: mean={mean_value:.6f}, std={std_value:.6f}\n")
    
    # Return in the expected format (list format for compatibility)
    return {"mean": [mean_value], "std": [std_value]}
