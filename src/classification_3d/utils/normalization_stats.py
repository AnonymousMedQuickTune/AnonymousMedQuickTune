import numpy as np
import torch
from monai.transforms import LoadImage

# TODO @Diane: Integrate is_rgb into run pipeline (low priority)
def autonorm(hyperparameters, is_rgb=False):  
    """
    Get normalization parameters from NePS hyperparameters.
    Values are normalized to be between 0 and 1.
    
    Args:
        hyperparameters (dict): NePS hyperparameters
        is_rgb (bool): Whether the dataset is RGB or grayscale. For RGB datasets we have 3 channels, for grayscale we have 1 channel.
    
    Returns:
        dict: Dictionary containing normalization parameters
    """
    # Use normalization stats from NePS hyperparameters
    print(f"\nNormalization parameters from NePS:")
    if is_rgb:
        print(f"\n\n\nDataset is RGB\n\n\n")
        mean_values = np.array(
            [
                float(hyperparameters["mean_1"]),
                float(hyperparameters["mean_2"]),
                float(hyperparameters["mean_3"]),
            ],
            dtype=np.float32,
        )
        std_values = np.array(
            [
                float(hyperparameters["std_1"]),
                float(hyperparameters["std_2"]),
                float(hyperparameters["std_3"]),
            ],
            dtype=np.float32,
        )
    else:
        print(f"\n\n\nDataset is grayscale\n\n\n")
        mean_values = np.array(
            [
                float(hyperparameters["mean_1"]),
            ],
            dtype=np.float32,
        )
        std_values = np.array(
            [
                float(hyperparameters["std_1"]),
            ],
            dtype=np.float32,
        )
    print(f"Mean: {mean_values}")
    print(f"Std: {std_values}\n")

    normalization_stats = {"mean": mean_values, "std": std_values}

    return normalization_stats

# TODO @Diane: Integrate is_rgb into run pipeline (low priority)
# TODO @Diane: Double check implementation of normalization stats
def calculate_normalization_stats(train_data, is_rgb=False): 
    """
    Calculate mean and standard deviation across all pixel values in all 3D volumes for each channel separately.
    Values are normalized to be between 0 and 1.
    
    Args:
        train_data (list): List of dictionaries containing 'image' key with file paths to 3D volumes
        is_rgb (bool): Whether the dataset is RGB or grayscale. For RGB datasets we have 3 channels, for grayscale we have 1 channel.
    
    Returns:
        dict: Dictionary containing normalization parameters with keys 'mean' and 'std'
    """
    # Initialize lists to store statistics for each volume
    all_means = []
    all_stds = []
    
    # Load and process each image
    loader = LoadImage(image_only=True)
    CLIP_MAX = 4095  # Maximum value for clipping and normalization (e.g., 4095 for 12-bit images, 255 for 8-bit images)
    
    for data_dict in train_data:
        # Load the image from file path
        img = loader(data_dict['image'])
        
        # Convert to torch tensor if it's not already
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        
        if is_rgb:
            raise NotImplementedError("RGB images are not supported yet!")
        
        else:
            # Add channel dimension if not present
            if img.ndim == 3:  # If shape is [H, W, D]
                img = img.unsqueeze(0)  # Add channel dimension -> [1, H, W, D]

            # Clip values to the specified maximum
            img = torch.clamp(img, 0, CLIP_MAX)
            
            # Normalize to [0, 1]
            img = img / CLIP_MAX
            # print(f"Min value: {torch.min(img).item()}, Max value: {torch.max(img).item()}")
            
            # Calculate mean and std for grayscale volume
            means = torch.mean(img, dim=[1, 2, 3])  # Mean across spatial dimensions [H, W, D]
            stds = torch.std(img, dim=[1, 2, 3])   # Std across spatial dimensions [H, W, D]
            
            all_means.append(means)
            all_stds.append(stds)
    
    # Stack the statistics (not the volumes)
    all_means = torch.stack(all_means)  # Shape: [N, 1]
    all_stds = torch.stack(all_stds)    # Shape: [N, 1]
    
    # Calculate final means and stds across all volumes
    means = torch.mean(all_means, dim=0)
    stds = torch.mean(all_stds, dim=0)

    # print(f"\nNormalization parameters (calculated):")
    # print(f"Means: {means.tolist()}")
    # print(f"Stds: {stds.tolist()}\n")
    return {"mean": means.tolist(), "std": stds.tolist()}
