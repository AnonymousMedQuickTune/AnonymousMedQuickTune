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



# Global variables
# These names can change depending in the dataset, format of the scans, segmentations, etc. 
# What else should be added as a global variable? Should this be part of the config file? 
IMAGE_NAME = "image.nii.gz"
SEGMENTATION_NAME = "segmentation.nii.gz"
MODALITY = "MRI"
MEDIAN_VOXEL = (0.68, 0.68, 5.0)

# Possible
DATASET_NAME = "lipo"

def load_3d_dataset(name, data_path="datasets", seed=42):
    """
    Load and preprocess a medical image dataset.

    Args:
        name (str): Name of the dataset to load ('lipo', 'desmoid', 'gist')
        data_path (str): Base path to the datasets directory. Defaults to 'datasets'

    Returns:
        dict: Dictionary containing dataset splits and metadata
    """

    # TODO: Implement 3D dataset loading

    images, segmentations, csv_path = get_paths(data_path, name) # Segmentation will be added in the next run. 

    # Load labels
    labels_csv = pd.read_csv(csv_path)
    labels = labels_csv['Diagnosis_binary'].to_numpy()

    # Filter out all samples with label -1 (e.g., invalid or insufficient class samples)
    # TODO @Natalia: This is a hack to get the dataset to work. We should find a better way to handle this.
    if name == "lipo":
        # Create a list of indices for which the label is not -1
        filtered_indices = [i for i, label in enumerate(labels) if label != -1]
        # Keep only the images corresponding to valid indices
        images = [images[i] for i in filtered_indices]
        # Keep only the segmentations corresponding to valid indices
        segmentations = [segmentations[i] for i in filtered_indices]
        # Keep only the labels that are not -1
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
    directory_names = sorted(os.listdir(full_path), key=natural_key)

    image_name = IMAGE_NAME
    segmentation_name = SEGMENTATION_NAME

    images_path = [os.path.join(full_path, d, image_name) for d in directory_names]
    segmentations_path = [os.path.join(full_path, d, segmentation_name) for d in directory_names]

    csv_path = os.path.join(full_path, "dataset.csv")

    return images_path, segmentations_path, csv_path

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def cache_datasets(name, data_path="datasets") -> None: # Preprocessed voxel size in the next run. This is not active as no cache is needed. 
    # Cache is used if there are multiple options of voxel size and they are calculated separately. If not, then just the trasnformations are needed.
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
        

def get_dataloaders(
    data,
    labels,
    k_folds,
    batch_size,
    num_workers,
    fold_idx,
    developer_mode=False,
):

    """
    Create data loaders for k-fold cross validation of brain tumor dataset.

    Args:
        data (list): Combined training and validation data
        labels (numpy.ndarray): Combined training and validation labels
        k_folds (int): Number of folds for cross-validation
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        fold_idx (int): Current fold index
        normalization_stats (dict, optional): Pre-computed normalization statistics
        augmentation_type (str): Type of augmentation to use

    Returns:
        tuple: (train_loader, val_loader) for the current fold
    """
    # Create output directory if it doesn't exist
    #Path(output_path).mkdir(parents=True, exist_ok=True)

    # Here is all the catched value of the paths for the specific voxel size. 

    #X_train, y_train, _, _, = load_3d_dataset(dataset_name)

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

    # First preprocess part:
    train_dataset = Dataset(train_data, transform=FullTransform(MEDIAN_VOXEL, developer_mode=developer_mode))
    val_dataset = Dataset(valid_data, transform=FullTransform(MEDIAN_VOXEL, developer_mode=developer_mode))
 
    # Try without cropping and padding if it is possible, if not need to add cropping.

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

def FullTransform(voxel, developer_mode=False):
    if developer_mode:
        target_shape = (64, 64, 32)  # Smaller shape for faster training on the laptop
    else:
        target_shape = (256, 256, 32)  # Original shape

    transforms = [
        LoadImaged(keys="image", image_only=True),  # Load NIfTI images
        EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
        Spacingd(keys="image", pixdim=voxel, mode="bilinear"),  # Resample to target spacing

        # Ensure all images have the same shape after resampling
        # TODO @Natalia: Won't this cause problems? + How to set target shape optimally?
        ResizeWithPadOrCropd(keys="image", spatial_size=target_shape),  # Pad or crop to fixed shape

        NormalizeIntensityd(keys=["image"]),
        RandFlipd( keys=["image"], prob=0.2, spatial_axis=0),
        RandRotated( keys=["image"], range_z=(-25, 25), prob=0.2),
        RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
    ]

    return Compose(transforms)