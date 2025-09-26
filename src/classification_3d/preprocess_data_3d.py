import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
import nibabel as nib
import shutil
import re

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

from src.classification_3d.utils.normalization_stats import calculate_normalization_stats
from src.classification_3d.preprocessing.preprocess import main_preprocessing
from src.classification_3d.preprocessing.utils import spacing_info
from src.classification_3d.utils.dataset_info import analyze_dataset_statistics, save_statistics_to_file, save_cv_split_info
from src.classification_3d.utils.dataset_cleaning import find_valid_image_and_segmentation_files, natural_key, clean_dataset
import datetime


def get_paths(dataset_path):
    """
    Get paths to images, segmentations, and CSV file for a given dataset path.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        tuple: Tuple containing lists of image paths, segmentation paths, and CSV file path
    """
    # Only directories, not files and ignore preprocessed directory which is within the cleaned dataset directory
    directory_names = [d for d in sorted(os.listdir(dataset_path), key=natural_key) 
                      if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith("preprocessed")]
    
    # Use flexible file naming to find image and segmentation files
    images_path = []
    segmentations_path = []
    
    for data_point in directory_names:
        # Supports: image.nii.gz, img.nii, image.nrrd, segmentation.nii.gz, mask.nii, etc.
        img_path, seg_path = find_valid_image_and_segmentation_files(dataset_path, data_point)
        if img_path and seg_path:
            images_path.append(img_path)
            segmentations_path.append(seg_path)
        else:
            print(f"\nWarning: Could not find image or segmentation files in {data_point}")
    
    csv_path = os.path.join(dataset_path, "dataset.csv")

    return images_path, segmentations_path, csv_path


# TODO @Natalia: Pls double check this implementation + compare calculated voxel size with values you worked with so far
# NOTE: Pls see experimental_setting.yaml > data.voxel_calculation
# NOTE: Pls see cleaned_dataset_path/preprocessed_*/statistics.txt
def calculate_voxel_size_from_images(cleaned_dataset_path, calculation_method="median"):
    """
    Calculate voxel for a dataset using the specified calculation method.
    
    Args:
        cleaned_dataset_path (str): Path to the cleaned dataset
        calculation_method (str): Method to calculate voxel size:
            - 'mean': Calculate mean voxel size across all training images
            - 'median': Calculate median voxel size across all training images
            - 'isotropic': Return (1.0, 1.0, 1.0)
            - 'volumetric_isotropic': Calculate isotropic voxel based on median volume
    
    Returns:
        tuple: Voxel size as (x, y, z) tuple
    """
    # Get image paths for the dataset
    images_path, _, _ = get_paths(cleaned_dataset_path)

    # Extract dataset name from path
    dataset_name = os.path.basename(cleaned_dataset_path).replace('_cleaned', '')

    # If isotropic, no calculation is needed, return (1.0, 1.0, 1.0)
    if calculation_method == "isotropic":
        voxel_result = (1.0, 1.0, 1.0)
        print(f"> Voxel size (isotropic) for {dataset_name} dataset: x={voxel_result[0]:.3f}, y={voxel_result[1]:.3f}, z={voxel_result[2]:.3f}")
        return voxel_result
    
    # Load voxel information from all images
    voxel_sizes = []
    volumes = []
    
    for i, img_path in enumerate(images_path):
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
        print(f"> Voxel size (mean) for {dataset_name} dataset: x={voxel_result[0]:.3f}, y={voxel_result[1]:.3f}, z={voxel_result[2]:.3f}")
        return voxel_result
    elif calculation_method == "median":
        # Calculate median across all images for each axis (x, y, z)
        # voxel_sizes shape: (N_images, 3_axes) -> axis=0 takes median over N_images for each axis
        voxel_result = tuple(np.median(voxel_sizes, axis=0))
        print(f"> Voxel size (median) for {dataset_name} dataset: x={voxel_result[0]:.3f}, y={voxel_result[1]:.3f}, z={voxel_result[2]:.3f}")
        return voxel_result
    elif calculation_method == "volumetric_isotropic":
        median_volume = np.median(volumes)
        # Calculate isotropic voxel size that gives the same volume
        isotropic_voxel = median_volume ** (1/3)
        voxel_result = (isotropic_voxel, isotropic_voxel, isotropic_voxel)
        print(f"> Voxel size (volumetric_isotropic) for {dataset_name} dataset: x={voxel_result[0]:.3f}, y={voxel_result[1]:.3f}, z={voxel_result[2]:.3f}")
        return voxel_result
    else:
        raise ValueError(f"Unknown calculation method: {calculation_method}")


def apply_smart_preprocessing(cleaned_dataset_path, voxel_size, calculation_method):
    """
    Apply Natalia's smart preprocessing pipeline to the dataset.
    
    Args:
        cleaned_dataset_path (str): Path to the cleaned dataset
        calculation_method (str): Method to calculate voxel size ('mean', 'median', 'isotropic', 'volumetric_isotropic')
        
    Returns:
        str: Path to the preprocessed dataset
    """
    print(f"\nApplying smart preprocessing to '{cleaned_dataset_path}'...\n")

    # Get image paths for the cleaned dataset
    images_path, segmentations_path, csv_path = get_paths(cleaned_dataset_path)
    
    # Extract directory names from image paths
    # Example: img_path = "datasets/lipo_cleaned/Lipo-001/image.nii.gz"
    #          os.path.dirname(img_path) → "datasets/lipo_cleaned/Lipo-001"
    #          os.path.basename(...) → "Lipo-001"
    # Result: ["Lipo-001", "Lipo-002", "Lipo-003", ...]
    directory_names = [os.path.basename(os.path.dirname(img_path)) for img_path in images_path]
    
    # Create file paths for preprocessing (only valid ones)
    # Example: cleaned_dataset_path = "datasets/lipo_cleaned"
    #          data_point = "Lipo-001"
    #          os.path.join(...) → "datasets/lipo_cleaned/Lipo-001"
    # Result: ["datasets/lipo_cleaned/Lipo-001", "datasets/lipo_cleaned/Lipo-002", ...]
    file_paths = [os.path.join(cleaned_dataset_path, data_point) for data_point in directory_names] 
    
    # Create preprocessed directory within the cleaned dataset directory
    output_path = os.path.join(cleaned_dataset_path, f"preprocessed_{calculation_method}")
    os.makedirs(output_path, exist_ok=True)
    
    # Run the preprocessing pipeline from Natalia's preprocessing code base
    main_preprocessing(file_paths, output_path, voxel_size)  # TODO @Natalia: Verify for correct integration pls :)
    
    # Analyze preprocessed dataset statistics
    print("\n=== Preprocessed Dataset Statistics Analysis ===")
    
    # Load the CSV file to get labels
    csv_path = os.path.join(cleaned_dataset_path, "dataset.csv")
    if os.path.exists(csv_path):
        labels_df = pd.read_csv(csv_path)
        statistics = analyze_dataset_statistics(output_path, labels_df)
        
        # Extract dataset name from path
        dataset_name = os.path.basename(cleaned_dataset_path).replace('_cleaned', '')
        
        # Save statistics to file in preprocessed directory
        statistics_file = os.path.join(output_path, "statistics.txt")
        additional_info = {
            "Voxel size": voxel_size,
            "Calculation method": calculation_method,
            "Preprocessing path": output_path
        }
        save_statistics_to_file(statistics, statistics_file, dataset_name, additional_info)
        
        print(f"Preprocessed statistics saved to: {statistics_file}")
    else:
        print("Warning: Could not find CSV file for statistics analysis")
    
    print(f"Smart preprocessing completed with {calculation_method} voxel calculation. Output saved to: {output_path}")
    return output_path, voxel_size


def load_3d_dataset(experiment_base_dir, dataset_name, data_path="datasets", seed=42, use_smart_preprocessing=True, voxel_calculation="median", cv_fold=0, mode="train"):
    """
    Load and preprocess a medical image dataset with cross-validation support.

    Args:
        experiment_base_dir (str): Path to the experiment base directory
        dataset_name (str): Name of the dataset to load ('lipo', 'desmoid', 'gist')
        data_path (str): Base path to the datasets directory. Defaults to 'datasets'
        seed (int): Random seed for reproducibility
        use_smart_preprocessing (bool): Whether to apply Natalia's smart preprocessing
        voxel_calculation (str): Method to calculate voxel size for preprocessing
        cv_fold (int): Cross-validation fold number (0, 1, 2, ...) for different train+val/test splits
        mode (str): Mode of the experiment ('train' or 'test')

    Returns:
        dict: Dictionary containing dataset splits and metadata
    """
    if use_smart_preprocessing:
        # Check if cleaned dataset exists
        cleaned_dataset_path = os.path.join(data_path, f"{dataset_name}_cleaned")
        if os.path.exists(cleaned_dataset_path) and os.path.exists(os.path.join(cleaned_dataset_path, "dataset.csv")):  
            print(f"> Found existing cleaned dataset at {cleaned_dataset_path}, skipping dataset cleaning...\n")
        else:
            print("\nX Cleaned dataset not found, running dataset cleaning...\n")
            cleaned_dataset_path = clean_dataset(data_path, dataset_name)

        # Check if preprocessed dataset with the given voxel calculation method exists
        preprocessed_dataset_path = os.path.join(cleaned_dataset_path, f"preprocessed_{voxel_calculation}")
        if os.path.exists(preprocessed_dataset_path):
            print(f"> Found existing preprocessed dataset at {preprocessed_dataset_path}, skipping preprocessing...\n")
            # Get voxel size from existing cleaned data (we'll calculate it again)
            voxel_size = calculate_voxel_size_from_images(cleaned_dataset_path, calculation_method=voxel_calculation)
        else:
            print("X Preprocessed dataset not found, running preprocessing...\n")
            voxel_size = calculate_voxel_size_from_images(cleaned_dataset_path, calculation_method=voxel_calculation)
            preprocessed_dataset_path, voxel_size = apply_smart_preprocessing(cleaned_dataset_path, voxel_size, calculation_method=voxel_calculation)
        # Keep the CSV path from the cleaned directory
        csv_path = os.path.join(cleaned_dataset_path, "dataset.csv")
    
    else:
        raise NotImplementedError("Smart preprocessing must be applied to use this function.")

    # Get image and segmentation paths from preprocessed data
    images, segmentations, _ = get_paths(preprocessed_dataset_path)

    # Load labels
    labels_csv = pd.read_csv(csv_path)
    labels = labels_csv['Diagnosis_binary'].to_numpy()

    # Filter out all samples with label -1 or NaN (e.g., invalid or insufficient class samples)
    # NOTE: When adding a new dataset, pls verify if -1 is not a valid label!
    if dataset_name in ["lipo", "desmoid", "gist"]:
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
    print(f"\nClass distribution after filtering: {dict(zip(unique_labels, counts))}")

    # Cross-validation for train+val/test splits
    # Use different seeds for different CV folds to get different splits
    cv_seed = seed + cv_fold
    
    # Split into train+val and test (80-20) with CV fold-specific seed
    test_size = 0.2
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=cv_seed, stratify=labels
    )

    print(f"\n> CV Fold {cv_fold}: Dataset split (train+val/test): {len(train_val_data)}/{len(test_data)} in a {int(100-(test_size*100))}%/{int(test_size*100)}% split\n")

    # Save CV split information to cv_summary folder
    cv_split_dir = os.path.join(experiment_base_dir, "cv_summary", "cv_splits")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    split_file = os.path.join(cv_split_dir, f"cv_fold_{cv_fold}_split_info_{timestamp}.txt")

    save_cv_split_info( 
        cv_split_dir,
        split_file,
        dataset_name, 
        cv_fold, 
        train_val_data, 
        test_data, 
        train_val_labels, 
        test_labels, 
        voxel_calculation,
        seed
    )
    
    # Save CV split in the preprocessed dataset folder
    cv_split_dir = os.path.join(preprocessed_dataset_path, "cv_splits")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    split_file = os.path.join(cv_split_dir, f"{mode}_{str(experiment_base_dir).split('/')[-2]}_{str(experiment_base_dir).split('/')[-1]}_cv_fold_{cv_fold}_split_info_{timestamp}.txt")
    
    save_cv_split_info( 
        cv_split_dir,
        split_file,
        dataset_name, 
        cv_fold, 
        train_val_data, 
        test_data, 
        train_val_labels, 
        test_labels, 
        voxel_calculation,
        seed
    )

    return {
        "train_val_data": train_val_data,  # TODO @Diane: rename to train_val_images
        "train_val_labels": train_val_labels,
        "test_images": test_data,  # TODO @Diane: rename to test_images
        "test_labels": test_labels,
        "num_classes": len(unique_labels),
        "voxel_size": voxel_size,
    }
    

def BasicAugmentTransform(voxel_size, normalization_stats, developer_mode):
    """
    Transform for training on the training set with basic data augmentation.
    
    Args:
        voxel_size (tuple): Voxel size in (x, y, z) format
        normalization_stats (dict): Normalization statistics
        developer_mode (bool): If True, uses smaller model target shape for faster development

    Returns:
        monai.transforms.Compose: Compose object containing the transformations
    """
    if developer_mode:
        spatial_size = (100, 100, 50)  # spatial_size in (H, W, D) format
        transforms = [
            LoadImaged(keys="image", image_only=True),  # Load NIfTI images
            EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
            Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
            # TODO @Diane: Double check if normalization stats are correctly used and calculated!
            # NormalizeIntensityd(keys=["image"], subtrahend=float(normalization_stats["mean"][0]), divisor=float(normalization_stats["std"][0])),
            NormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=1.0),
            
            # NOTE: Use smaller image size in the developer mode for faster development!
            ResizeWithPadOrCropd(keys="image", spatial_size=spatial_size, mode="constant", constant_values=0),

            # Data augmentation  # TODO @Diane: improve data augmentation strategy + add hyperparameters to the search space
            RandFlipd( keys=["image"], prob=0.2, spatial_axis=0),
            RandRotated( keys=["image"], range_z=(-25, 25), prob=0.2),
            RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
        ]
    else:
        # TODO @Natalia: Delete spatial_size after model is able to handle different input sizes!
        # Please see statistics.txt in lipo_cleaned/preprocessed_*/statistics.txt for the maximum width, height, and depth.
        # For preprocessed_mean, the maximum width, height, and depth are 466, 558, and 50 respectively.
        # For preprocessed_median, the maximum width, height, and depth are 446, 534, and 176 respectively.
        # For preprocessed_isotropic, the maximum width, height, and depth are 381, 382, and 242 respectively.
        # For preprocessed_volumetric_isotropic, the maximum width, height, and depth are 274, 275, and 176 respectively.
        # Use approximate comparison with tolerance for floating point precision issues
        if abs(voxel_size[0] - 0.68684727) < 1e-6:  # mean voxel calculation
            spatial_size = (466, 558, 50)  # spatial_size in (H, W, D) format
        elif abs(voxel_size[0] - 0.71651787) < 1e-6:  # median voxel calculation
            spatial_size = (446, 534, 50)  # spatial_size in (H, W, D) format
        elif abs(voxel_size[0] - 1.0) < 1e-6:  # isotropic voxel calculation
            spatial_size = (381, 382, 176)  # spatial_size in (H, W, D) format
        elif abs(voxel_size[0] - 1.3903084893330422) < 1e-6:  # volumetric isotropic voxel calculation
            spatial_size = (274, 275, 176)  # spatial_size in (H, W, D) format
        else:
            print(f"Warning: Unknown voxel_size[0] = {voxel_size[0]}, using default spatial_size")
        
        transforms = [
            LoadImaged(keys="image", image_only=True),  # Load NIfTI images
            EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
            Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
            # TODO @Diane: Double check if normalization stats are correctly used and calculated!
            # NormalizeIntensityd(keys=["image"], subtrahend=float(normalization_stats["mean"][0]), divisor=float(normalization_stats["std"][0])),
            NormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=1.0),
            
            
            # Neither DenseNetV1 nor DenseNetV2 model can handle variable size inputs. # TODO @Natalia: Please check this out
            # We use pad to reach the maximum sizes to make the model work for now.
            # NOTE: This is hardcoded for lipo dataset!!
            # TODO @Natalia: Delete after model is able to handle different input sizes!
            ResizeWithPadOrCropd(keys="image", spatial_size=spatial_size, mode="constant", constant_values=0),

            # Data augmentation  # TODO @Diane: Improve data augmentation strategy + add hyperparameters to the search space
            RandFlipd( keys=["image"], prob=0.2, spatial_axis=0),
            RandRotated( keys=["image"], range_z=(-25, 25), prob=0.2),
            RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
        ]

    return Compose(transforms)


def EvaluationTransform(voxel_size, normalization_stats, developer_mode):
    """
    Transform for evaluation on validation and test set without data augmentation.

    Args:
        voxel_size (tuple): Voxel size in (x, y, z) format
        normalization_stats (dict): Normalization statistics
        developer_mode (bool): If True, uses smaller model target shape for faster development

    Returns:
        monai.transforms.Compose: Compose object containing the transformations
    """
    if developer_mode:
        spatial_size = (100, 100, 50)  # spatial_size in (H, W, D) format
        transforms = [
            LoadImaged(keys="image", image_only=True),  # Load NIfTI images
            EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
            Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
            # TODO @Diane: Double check if normalization stats are correctly used and calculated!
            # NormalizeIntensityd(keys=["image"], subtrahend=float(normalization_stats["mean"][0]), divisor=float(normalization_stats["std"][0])),
            NormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=1.0),

            # NOTE: Use smaller image size in the developer mode for faster development!
            ResizeWithPadOrCropd(keys="image", spatial_size=spatial_size, mode="constant", constant_values=0),

            # No data augmentation for evaluation!
        ]
    else:
        # TODO @Natalia: Delete spatial_size after model is able to handle different input sizes!
        # Please see statistics.txt in lipo_cleaned/preprocessed_*/statistics.txt for the maximum width, height, and depth.
        # For preprocessed_mean, the maximum width, height, and depth are 466, 558, and 50 respectively.
        # For preprocessed_median, the maximum width, height, and depth are 446, 534, and 176 respectively.
        # For preprocessed_isotropic, the maximum width, height, and depth are 381, 382, and 242 respectively.
        # For preprocessed_volumetric_isotropic, the maximum width, height, and depth are 274, 275, and 176 respectively.
        # Use approximate comparison with tolerance for floating point precision issues
        if abs(voxel_size[0] - 0.68684727) < 1e-6:  # mean voxel calculation
            spatial_size = (466, 558, 50)  # spatial_size in (H, W, D) format
        elif abs(voxel_size[0] - 0.71651787) < 1e-6:  # median voxel calculation
            spatial_size = (446, 534, 50)  # spatial_size in (H, W, D) format
        elif abs(voxel_size[0] - 1.0) < 1e-6:  # isotropic voxel calculation
            spatial_size = (381, 382, 176)  # spatial_size in (H, W, D) format
        elif abs(voxel_size[0] - 1.3903084893330422) < 1e-6:  # volumetric isotropic voxel calculation
            spatial_size = (274, 275, 176)  # spatial_size in (H, W, D) format
        else:
            print(f"Warning: Unknown voxel_size[0] = {voxel_size[0]}, using default spatial_size")

        transforms = [
            LoadImaged(keys="image", image_only=True),  # Load NIfTI images
            EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
            Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
            # TODO @Diane: Double check if normalization stats are correctly used and calculated!
            # NormalizeIntensityd(keys=["image"], subtrahend=float(normalization_stats["mean"][0]), divisor=float(normalization_stats["std"][0])),
            NormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=1.0),

            # Neither DenseNetV1 nor DenseNetV2 model can handle variable size inputs. # TODO @Natalia: Please check this out
            # We use pad to reach the maximum sizes to make the model work for now.
            # NOTE: This is hardcoded for lipo dataset!!
            # TODO @Natalia: Delete after model is able to handle different input sizes!
            ResizeWithPadOrCropd(keys="image", spatial_size=spatial_size, mode="constant", constant_values=0),

            # No data augmentation for evaluation!
        ]
        
    return Compose(transforms)

def get_kfold_dataloaders(
    dataset_name,
    data,
    labels,
    k_folds,
    batch_size,
    num_workers,
    fold_idx,
    voxel_size,
    normalization_stats,
    augmentation_type,
    developer_mode,
    no_validation=False,
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
        no_validation (bool): If True, uses no validation set

    Returns:
        tuple: (train_loader, val_loader) for the current fold
    """
    # Handle no_validation case
    if no_validation:
        print("No validation set mode: Using all data for training")
        # Use all data for training, no validation split
        train_data_images = [{"index": idx, "image": img, "label": label} 
                        for idx, (img, label) in enumerate(zip(data, labels))]
        train_data = train_data_images
        valid_data = []  # Empty validation set
    else:
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

    # Calculate normalization stats from preprocessed training data if not provided by NePS
    # NOTE: normalization stats should be calculated from the preprocessed data, not the original data!
    if normalization_stats is None:
        # TODO @Diane: Verify this implementation!
        stats = calculate_normalization_stats(train_data)  
        normalization_stats = {"mean": stats["mean"], "std": stats["std"]}
        print(f"Normalization stats (calculated from preprocessed data):\n{normalization_stats}     !!! Currently ignored > see applied data transformations !!!\n")

    # Create train and validation datasets
    if augmentation_type == "basic":
        train_dataset = Dataset(train_data, transform=BasicAugmentTransform(voxel_size, normalization_stats, developer_mode))
    elif augmentation_type == "trivial":
        raise NotImplementedError("Trivial augmentation is not implemented yet.")  # TODO @Diane: Integrate TrivialAugment
    elif augmentation_type == "groupaugment":
        raise NotImplementedError("Group augmentation is not implemented yet.")  # TODO @Diane: Implement + integrate GroupAugment
    else:
        raise ValueError(f"Invalid augmentation type: {augmentation_type}")

    # Create validation dataset only if validation data exists
    if valid_data:
        val_dataset = Dataset(valid_data, transform=EvaluationTransform(voxel_size, normalization_stats, developer_mode))
    else:
        val_dataset = None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Create empty validation loader when no validation data
        val_loader = None

    return train_loader, val_loader
