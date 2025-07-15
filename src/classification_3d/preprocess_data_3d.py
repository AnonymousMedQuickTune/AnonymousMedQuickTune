import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    RandRotated,
    RandZoomd,
    RandFlipd,
    ResizeWithPadOrCropd,
)
from monai.data import Dataset
from torch.utils.data import DataLoader
import re

from src.classification_3d.utils.normalization_stats import calculate_normalization_stats

def load_3d_dataset(name, data_path="datasets", seed=42):
    """
    Load and preprocess a medical image dataset.

    Args:
        name (str): Name of the dataset to load ('lipo', 'desmoid', 'gist')
        data_path (str): Base path to the datasets directory. Defaults to 'datasets'

    Returns:
        dict: Dictionary containing dataset splits and metadata
    """
    images, segmentations, csv_path = get_paths(data_path, name) # Segmentation will be added in the next run. 

    # Load labels
    labels_csv = pd.read_csv(csv_path)
    labels = labels_csv['Diagnosis_binary'].to_numpy()

    # Filter out all samples with label -1 or NaN (e.g., invalid or insufficient class samples)
    # NOTE: When adding a new dataset, pls verify if -1 is not a valid label!
    if name in ["lipo", "desmoid", "gist"]:
        # Create a list of indices for which the label is not -1 and not NaN
        filtered_indices = [i for i, label in enumerate(labels) if label != -1 and not pd.isna(label)]
        # Keep only the images corresponding to valid indices
        images = [images[i] for i in filtered_indices]
        # Keep only the segmentations corresponding to valid indices
        segmentations = [segmentations[i] for i in filtered_indices]
        # Keep only the labels that are not -1 and not NaN
        labels = [labels[i] for i in filtered_indices]

    # Recheck class distribution after filtering
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution after filtering: {dict(zip(unique_labels, counts))}")

    # Split into train+val and test (80-20)
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    print(f"\nDataset split (train+val/test): {len(train_val_data)}/{len(test_data)}")

    return {
        "train_val_data": train_val_data,
        "train_val_labels": train_val_labels,
        "test_data": test_data,
        "test_labels": test_labels,
        "num_classes": len(unique_labels),
    }
    
def get_paths(data_path, name):
    full_path = os.path.join(data_path, name)
    # Only directories, not files
    directory_names = [d for d in sorted(os.listdir(full_path), key=natural_key) 
                      if os.path.isdir(os.path.join(full_path, d))]
    
    if name in ["lipo", "desmoid", "gist"]:
        # These names can change depending in the dataset, format of the scans, segmentations, etc. 
        # Please verify this when adding new datasets.
        image_name = "image.nii.gz"
        segmentation_name = "segmentation.nii.gz"
    else:
        raise NotImplementedError(f"Filename for dataset {name} not specified yet.")

    images_path = [os.path.join(full_path, d, image_name) for d in directory_names]
    segmentations_path = [os.path.join(full_path, d, segmentation_name) for d in directory_names]

    csv_path = os.path.join(full_path, "dataset.csv")

    return images_path, segmentations_path, csv_path

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def cache_datasets(name, data_path="datasets") -> None: # Preprocessed voxel size in the next run. This is not active as no cache is needed. 
    # Cache is used if there are multiple options of voxel size and they are calculated separately. If not, then just the trasnformations are needed.
    # NOTE: Outcome of our last meeting: Use Natalia's smart preprocessing instead of this ResizeWithPadOrCropd > caching is needed
    # TODO @Natalia: Integrate your smart preprocessing to the cache_datasets function
    # TODO @Both: Discuss in a meeting: Can't we not modify the model to handle variable size inputs instead of using the smart preprocessing?
    
    """
    Preprocess and cache brain tumor datasets for faster experiment initialization.

    Args:
        config (DictConfig): Hydra configuration object
    """
    print("\nPreprocessing dataset...")

    # First, process the raw dataset
    raw_dataset_path = os.path.join(data_path + name, "/raw") # If there is a config path this could be changed. 
    processed_dataset_path = os.path.join(data_path + name, "/cache")

    if not os.path.exists(os.path.join(raw_dataset_path + "cache")): # If using voxel size for preprocessing this will change
        print("Processing raw dataset...")
        # get_dataloaders(raw_dataset_path, processed_dataset_path)
    else:
        print("Raw dataset already processed, skipping...")
        
def get_kfold_dataloaders(
    dataset_name,
    data,
    labels,
    k_folds,
    batch_size,
    num_workers,
    fold_idx,
    normalization_stats=None,
    augmentation_type="medical",
    developer_mode=False,
    voxel_calculation="median",
    data_path="datasets",
):
    """
    Create data loaders for k-fold cross validation of brain tumor dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'lipo', 'desmoid', 'gist')
        data (list): Combined training and validation data
        labels (numpy.ndarray): Combined training and validation labels
        k_folds (int): Number of folds for cross-validation
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        fold_idx (int): Current fold index
        normalization_stats (dict, optional): Pre-computed normalization statistics
        augmentation_type (str): Type of augmentation to use
        developer_mode (bool): If True, uses smaller model target shape for faster development
        voxel_calculation (str): Method to calculate voxel size
        data_path (str): Path to datasets directory

    Returns:
        tuple: (train_loader, val_loader) for the current fold
    """
    # Get voxel size depending on the dataset and calculation method
    voxel_size = calculate_voxel_from_images(data_path, dataset_name, calculation_method=voxel_calculation)

    # Create k-fold splitter
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Get indices for current fold
    indices = np.arange(len(data))
    for i, (train_idx, val_idx) in enumerate(kfold.split(indices, labels)):
        if i == fold_idx:
            break
    
    # Combine images and labels into a list of dictionaries
    train_data_images = [{"index": idx, "image": img, "label": label} 
                    for idx, (img, label) in enumerate(zip(data, labels))]
 
    # Split data for current fold
    train_data = [train_data_images[i] for i in train_idx]
    valid_data = [train_data_images[i] for i in val_idx]

    # TODO @Diane: Use resized data (pad or crop) to pass to the normalization stats function
    # NOTE: normalization stats should be calculated from resized (pad or crop) images, meaning the preprocessed data, not original images!!!!
    # Calculate normalization stats from training data if not provided
    if normalization_stats is None:
        means, stds = calculate_normalization_stats(train_data)
        normalization_stats = {"mean": means, "std": stds}

    # First preprocess part:  # TODO @Diane: integrate normalization stats and data augmentation to Dataset class
    train_dataset = Dataset(train_data, transform=FullTransform(voxel_size, developer_mode=developer_mode))
    val_dataset = Dataset(valid_data, transform=FullTransform(voxel_size, developer_mode=developer_mode))
 
    # Try without cropping and padding if it is possible, if not need to add cropping.  # NOTE: See FullTransform function

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

def FullTransform(voxel_size, developer_mode=False):
    if developer_mode:
        target_shape = (32, 32, 16)  # Smaller shape for faster training on the laptop
    else:
        target_shape = (256, 256, 32)  # Original shape
        # TODO @Natalia: How to set target shape optimally?

    transforms = [
        LoadImaged(keys="image", image_only=True),  # Load NIfTI images
        EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
        Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing

        # Ensure all images have the same shape
        # TODO: @Diane: Remove ResizeWithPadOrCropd and use Natalia's preprocessing instead (when it's ready to use)
        # NOTE: meeting outcome: Downsampling bad for images with small tumors and with ResizeWithPadOrCropd we might accidentally crop out the tumor
        # So we need to use Natalia's smarter preprocessing instead.
        # TODO @Both: Think about modifying the model to handle variable size inputs.
        ResizeWithPadOrCropd(keys="image", spatial_size=target_shape),  # Pad or crop to fixed shape

        NormalizeIntensityd(keys=["image"]),
        RandFlipd( keys=["image"], prob=0.2, spatial_axis=0),
        RandRotated( keys=["image"], range_z=(-25, 25), prob=0.2),
        RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
    ]

    return Compose(transforms)

# TODO @Natalia: Pls double check this implementation by trying out different calculation methods (see experimental_setting.yaml > data.voxel_calculation)
def calculate_voxel_from_images(data_path, dataset_name, calculation_method="median"):
    """
    Calculate voxel for a dataset using the specified calculation method.
    
    Args:
        dataset_name (str): Name of the dataset
        data_path (str): Path to datasets directory
        calculation_method (str): Method to calculate voxel size:
            - 'mean': Calculate mean voxel size across all training images
            - 'median': Calculate median voxel size across all training images
            - 'isotropic': Return (1.0, 1.0, 1.0)
            - 'volumetric_isotropic': Calculate isotropic voxel based on median volume
    
    Returns:
        tuple: Voxel size as (x, y, z) tuple
    """
    # Get image paths for the dataset
    images_path, _, _ = get_paths(data_path, dataset_name)

    # If isotropic, no calculation is needed, return (1.0, 1.0, 1.0)
    if calculation_method == "isotropic":
        voxel_result = (1.0, 1.0, 1.0)
        print(f"Voxel size (isotropic) for {dataset_name}: x={voxel_result[0]:.3f}, y={voxel_result[1]:.3f}, z={voxel_result[2]:.3f}")
        return voxel_result
    
    # Load voxel information from all images
    voxel_sizes = []
    volumes = []
    
    for img_path in images_path:
        try:
            # Load image header to get voxel information
            import nibabel as nib
            img = nib.load(img_path)
            voxel_size = img.header.get_zooms()[:3]  # Get first 3 dimensions
            voxel_sizes.append(voxel_size)
            
            # Calculate volume for volumetric_isotropic
            if calculation_method == "volumetric_isotropic":
                volume = np.prod(voxel_size)
                volumes.append(volume)
                
        except Exception as e:
            print(f"Warning: Could not load voxel information from {img_path}: {e}")
            continue
    
    if not voxel_sizes:
        raise ValueError(f"No valid voxel information found in images")
    
    voxel_sizes = np.array(voxel_sizes)
    
    if calculation_method == "mean":
        # Calculate mean across all images for each axis (x, y, z)
        # voxel_sizes shape: (N_images, 3_axes) -> axis=0 averages over N_images for each axis
        voxel_result = tuple(np.mean(voxel_sizes, axis=0))
        print(f"Voxel size (mean) for {dataset_name}: x={voxel_result[0]:.3f}, y={voxel_result[1]:.3f}, z={voxel_result[2]:.3f}")
        return voxel_result
    elif calculation_method == "median":
        # Calculate median across all images for each axis (x, y, z)
        # voxel_sizes shape: (N_images, 3_axes) -> axis=0 takes median over N_images for each axis
        voxel_result = tuple(np.median(voxel_sizes, axis=0))
        print(f"Voxel size (median) for {dataset_name}: x={voxel_result[0]:.3f}, y={voxel_result[1]:.3f}, z={voxel_result[2]:.3f}")
        return voxel_result
    elif calculation_method == "volumetric_isotropic":
        median_volume = np.median(volumes)
        # Calculate isotropic voxel size that gives the same volume
        isotropic_voxel = median_volume ** (1/3)
        voxel_result = (isotropic_voxel, isotropic_voxel, isotropic_voxel)
        print(f"Voxel size (volumetric_isotropic) for {dataset_name}: x={voxel_result[0]:.3f}, y={voxel_result[1]:.3f}, z={voxel_result[2]:.3f}")
        return voxel_result
    else:
        raise ValueError(f"Unknown calculation method: {calculation_method}")