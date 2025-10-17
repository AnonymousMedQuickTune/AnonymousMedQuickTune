import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import nibabel as nib
import shutil
import re
import pickle

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


def apply_smart_preprocessing(cleaned_dataset_path, voxel_size, calculation_method, is_mri):
    """
    Apply Natalia's smart preprocessing pipeline to the dataset.
    
    Args:
        cleaned_dataset_path (str): Path to the cleaned dataset
        voxel_size (tuple): Voxel size in (x, y, z) format
        calculation_method (str): Method to calculate voxel size ('mean', 'median', 'isotropic', 'volumetric_isotropic')
        is_mri (bool): Whether the dataset is MRI > MRI datasets need normalization in the preprocessing for each image individually
        
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
    main_preprocessing(file_paths, output_path, voxel_size, is_mri)  # TODO @Natalia: Verify for correct integration pls :)
    
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


def load_3d_dataset_with_outer_cv_splits(experiment_base_dir, dataset_name, data_path="datasets", seed=42, use_smart_preprocessing=True, voxel_calculation="median", cv_outer_fold=0, mode="train", cv_outer_folds_repeats=5, cv_outer_folds_splits=3):
    """
    Load and preprocess a medical image dataset with N-repeated K-fold stratified cross-validation.
    Automatically checks for existing CV splits and creates them if they don't exist.

    Args:
        experiment_base_dir (str): Path to the experiment base directory
        dataset_name (str): Name of the dataset to load ('lipo', 'desmoid', 'gist')
        data_path (str): Base path to the datasets directory. Defaults to 'datasets'
        seed (int): Random seed for reproducibility
        use_smart_preprocessing (bool): Whether to apply Natalia's smart preprocessing
        voxel_calculation (str): Method to calculate voxel size for preprocessing
        cv_outer_fold (int): Cross-validation fold number (0, 1, 2, ...) for different train+val/test splits
        mode (str): Mode of the experiment ('train' or 'test')
        cv_outer_folds_repeats (int): Number of repeats for repeated stratified K-fold
        cv_outer_folds_splits (int): Number of splits per repeat

    Returns:
        dict: Dictionary containing dataset splits and metadata
    """
    if use_smart_preprocessing:
        if dataset_name in ["lipo", "desmoid", "liver"]:
            is_mri = True
        elif dataset_name in ["gist", "crlm", "melanoma"]:
            is_mri = False  # CT dataset
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. If you want to add a new dataset, please add it to the list of MRI datasets or CT datasets.")
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
            preprocessed_dataset_path, voxel_size = apply_smart_preprocessing(cleaned_dataset_path, voxel_size, calculation_method=voxel_calculation, is_mri=is_mri)
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

    # STEP 1: Check if CV splits already exist
    cv_splits_dir = os.path.join(experiment_base_dir, "cv_splits")
    splits_file = os.path.join(cv_splits_dir, f"cv_splits_base_seed_{seed}_repeats_{cv_outer_folds_repeats}_splits_{cv_outer_folds_splits}.pkl")
    cv_splits = None
    print(f"\nCV splits:\n----------")

    if os.path.exists(splits_file):
        print(f"> Loading existing CV splits from: {splits_file}")
        with open(splits_file, "rb") as f:
            splits_data = pickle.load(f)
        cv_splits = splits_data["cv_splits"]
    else:
        print(f"> No existing CV splits found. Generating new ones...")
        
        # STEP 2: Generate CV splits using RepeatedStratifiedKFold
        rskf = RepeatedStratifiedKFold(
            n_splits=cv_outer_folds_splits, 
            n_repeats=cv_outer_folds_repeats, 
            random_state=seed  # Base seed - RepeatedStratifiedKFold generates different seeds internally
        )
        
        # Generate all splits
        cv_splits = list(rskf.split(images, labels))
        
        # Save CV splits to file
        os.makedirs(cv_splits_dir, exist_ok=True)
        splits_data = {
            "cv_splits": cv_splits,
            "n_repeats": cv_outer_folds_repeats,
            "n_splits": cv_outer_folds_splits,
            "base_seed": seed,
            "total_splits": len(cv_splits),
            "dataset_info": {
                "total_samples": len(images),
                "n_classes": len(np.unique(labels)),
                "class_distribution": {str(i): int(np.sum(np.array(labels) == i)) for i in np.unique(labels)}
            }
        }
        
        with open(splits_file, "wb") as f:
            pickle.dump(splits_data, f)
        
        print(f"> Generated {len(cv_splits)} CV splits using RepeatedStratifiedKFold (N={cv_outer_folds_repeats} * {cv_outer_folds_splits} folds)")
        print(f"> CV splits saved to: {splits_file}")
    
    # STEP 3: Use the CV splits for current fold
    if cv_outer_fold < len(cv_splits):
        train_indices, test_indices = cv_splits[cv_outer_fold]
        
        # Split data using the pre-generated indices
        train_val_images = [images[i] for i in train_indices]
        test_images = [images[i] for i in test_indices]
        train_val_labels = [labels[i] for i in train_indices]
        test_labels = [labels[i] for i in test_indices]
        
        print(f"\n> CV Fold {cv_outer_fold}: Dataset split (train+val/test): {len(train_val_images)}/{len(test_images)} using pre-generated CV splits")
    else:
        raise ValueError(f"CV fold {cv_outer_fold} >= number of available splits {len(cv_splits)}")

    # Save CV split information to cv_summary folder
    cv_split_dir = os.path.join(experiment_base_dir, "cv_summary", "cv_splits")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    split_file = os.path.join(cv_split_dir, f"cv_outer_fold_{cv_outer_fold}_split_info_{timestamp}.txt")

    save_cv_split_info( 
        cv_split_dir,
        split_file,
        dataset_name, 
        cv_outer_fold, 
        train_val_images, 
        test_images, 
        train_val_labels, 
        test_labels, 
        voxel_calculation,
        seed
    )
    
    # Save CV split in the preprocessed dataset folder
    cv_split_dir = os.path.join(preprocessed_dataset_path, "cv_splits")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    split_file = os.path.join(cv_split_dir, f"{mode}_{str(experiment_base_dir).split('/')[-2]}_{str(experiment_base_dir).split('/')[-1]}_cv_outer_fold_{cv_outer_fold}_split_info_{timestamp}.txt")
    
    save_cv_split_info( 
        cv_split_dir,
        split_file,
        dataset_name, 
        cv_outer_fold, 
        train_val_images, 
        test_images, 
        train_val_labels, 
        test_labels, 
        voxel_calculation,
        seed
    )

    return {
        "train_val_images": train_val_images,
        "train_val_labels": train_val_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "num_classes": len(unique_labels),
        "voxel_size": voxel_size,
    }
    

def BasicAugmentTransform(voxel_size, image_size, normalization_stats, developer_mode):
    """
    Transform for training on the training set with basic data augmentation.
    
    Args:
        voxel_size (tuple): Voxel size in (x, y, z) format
        image_size (tuple): Image size in (H, W, D) format for ViT; default None
        normalization_stats (dict): Normalization statistics
        developer_mode (bool): If True, uses smaller model target shape for faster development

    Returns:
        monai.transforms.Compose: Compose object containing the transformations
    """
    # TODO @Diane: improve data augmentation strategy + add hyperparameters to the search space
    if normalization_stats is None:
        # MRI Images: gist, crlm, melanoma
        # NOTE: Normalization is done in the preprocessing per image/patient individually
        if developer_mode or image_size is not None:
            transforms = [
                LoadImaged(keys="image", image_only=True),  # Load NIfTI images
                EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
                Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
                
                # NOTE: Use smaller image size in the developer mode for faster development!
                # NOTE: Use special image size for some models
                ResizeWithPadOrCropd(keys="image", spatial_size=image_size, mode="constant", constant_values=0),

                # Data augmentation  
                RandFlipd( keys=["image"], prob=0.2, spatial_axis=0),
                RandRotated( keys=["image"], range_z=(-25, 25), prob=0.2),
                RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
            ]
        else:
            transforms = [
                LoadImaged(keys="image", image_only=True),  # Load NIfTI images
                EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
                Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing

                # Data augmentation
                RandFlipd( keys=["image"], prob=0.2, spatial_axis=0),
                RandRotated( keys=["image"], range_z=(-25, 25), prob=0.2),
                RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
            ]
    else:
        # CT Images: lipo, desmoid, liver
        # NOTE: Normalization is done in the runpipeline based on training data statistics depending on the cross-validation folds.
        if developer_mode or image_size is not None:
            transforms = [
                LoadImaged(keys="image", image_only=True),  # Load NIfTI images
                EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
                Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
                NormalizeIntensityd(keys=["image"], subtrahend=normalization_stats["mean"][0], divisor=normalization_stats["std"][0]),
                
                # NOTE: Use smaller image size in the developer mode for faster development!
                # NOTE: Use special image size for some models
                ResizeWithPadOrCropd(keys="image", spatial_size=image_size, mode="constant", constant_values=0),

                # Data augmentation#
                RandFlipd( keys=["image"], prob=0.2, spatial_axis=0),
                RandRotated( keys=["image"], range_z=(-25, 25), prob=0.2),
                RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
            ]
        else:
            transforms = [
                LoadImaged(keys="image", image_only=True),  # Load NIfTI images
                EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
                Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
                NormalizeIntensityd(keys=["image"], subtrahend=normalization_stats["mean"][0], divisor=normalization_stats["std"][0]),

                # Data augmentation
                RandFlipd( keys=["image"], prob=0.2, spatial_axis=0),
                RandRotated( keys=["image"], range_z=(-25, 25), prob=0.2),
                RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
            ]

    return Compose(transforms)


def EvaluationTransform(voxel_size, image_size, normalization_stats, developer_mode):
    """
    Transform for evaluation on validation and test set without data augmentation.

    Args:
        voxel_size (tuple): Voxel size in (x, y, z) format
        image_size (tuple): Image size in (H, W, D) format for ViT; default None
        normalization_stats (dict): Normalization statistics
        developer_mode (bool): If True, uses smaller model target shape for faster development

    Returns:
        monai.transforms.Compose: Compose object containing the transformations
    """
    if normalization_stats is None:
        # MRI Images: gist, crlm, melanoma
        # NOTE: Normalization is done in the preprocessing per image/patient individually
        if developer_mode or image_size is not None:
            transforms = [
                LoadImaged(keys="image", image_only=True),  # Load NIfTI images
                EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
                Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing

                # NOTE: Use smaller image size in the developer mode for faster development!
                # NOTE: Use special image size for some models
                ResizeWithPadOrCropd(keys="image", spatial_size=image_size, mode="constant", constant_values=0),

                # No data augmentation for evaluation!
            ]
        else:
            transforms = [
                LoadImaged(keys="image", image_only=True),  # Load NIfTI images
                EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
                Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing

                # No data augmentation for evaluation!
            ]
    else:
        # CT Images: lipo, desmoid, liver
        # NOTE: Normalization is done in the runpipeline based on training data statistics depending on the cross-validation folds.
        if developer_mode or image_size is not None:
            transforms = [
                LoadImaged(keys="image", image_only=True),  # Load NIfTI images
                EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
                Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
                NormalizeIntensityd(keys=["image"], subtrahend=float(normalization_stats["mean"][0]), divisor=float(normalization_stats["std"][0])),

                # NOTE: Use smaller image size in the developer mode for faster development!
                # NOTE: Use special image size for some models
                ResizeWithPadOrCropd(keys="image", spatial_size=image_size, mode="constant", constant_values=0),

                # No data augmentation for evaluation!
            ]
        else:
            transforms = [
                LoadImaged(keys="image", image_only=True),  # Load NIfTI images
                EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
                Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
                NormalizeIntensityd(keys=["image"], subtrahend=float(normalization_stats["mean"][0]), divisor=float(normalization_stats["std"][0])),

                # No data augmentation for evaluation!
            ]
        
    return Compose(transforms)

def get_kfold_dataloaders(
    seed,
    dataset_name,
    data,
    labels,
    cv_inner_folds,
    batch_size,
    num_workers,
    fold_idx,
    voxel_size,
    normalization_stats,
    augmentation_type,
    developer_mode,
    image_size=None,
    fold_directory=None,
    no_validation=False,
):
    """
    Create data loaders for k-fold cross validation of brain tumor dataset.

    Args:
        seed (int): Random seed for reproducibility
        dataset_name (str): Name of the dataset (e.g., 'lipo', 'desmoid', 'gist')
        data (list): Combined training and validation data
        labels (numpy.ndarray): Combined training and validation labels
        cv_inner_folds (int): Number of folds for cross-validation
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        fold_idx (int): Current fold index
        voxel_size (tuple): Voxel size for the dataset
        normalization_stats (dict, optional): Pre-computed normalization statistics
        augmentation_type (str): Type of augmentation to use
        developer_mode (bool): If True, uses smaller model target shape for faster development
        image_size (tuple): Image size in (H, W, D) format for ViT; default None
        fold_directory (str, optional): Directory path for saving normalization stats
        no_validation (bool): If True, does not split train data in to train/val splits for validation
        
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
        kfold = StratifiedKFold(n_splits=cv_inner_folds, shuffle=True, random_state=seed)

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

    # Calculate normalization stats from preprocessed CT training data if not provided by NePS (AutoNorm)
    if dataset_name in ["lipo", "desmoid", "liver"]:  # MRI datasets
        # Normalization is done in the preprocessing per image/patient individually
        normalization_stats = None
    elif dataset_name in ["gist", "crlm", "melanoma"]:  # CT datasets
        if normalization_stats is None:
            # Calculate normalization stats from preprocessed training data if not provided by NePS (AutoNorm)
            # The training data is dependent on the cross-validation fold.
            stats = calculate_normalization_stats(train_data)  
            normalization_stats = {"mean": stats["mean"], "std": stats["std"]}
            stats_source = "Calculated from training data"
        else:
            # Normalization stats provided by NePS (AutoNorm)
            print(f"Normalization stats provided by NePS (AutoNorm): {normalization_stats}")
            stats_source = "AutoNorm (NePS)"
        
        # Save normalization stats to a file in the directory of the inner CV fold
        if fold_directory is not None:
            normalization_stats_file = os.path.join(fold_directory, "normalization_stats.txt")
            with open(normalization_stats_file, "w", encoding="utf-8") as f:
                f.write(f"Normalization Statistics for Inner CV Fold {fold_idx}\n")
                f.write(f"{'='*50}\n")
                f.write(f"Mean: {normalization_stats['mean']}\n")
                f.write(f"Std:  {normalization_stats['std']}\n")
                f.write(f"\nSource: {stats_source}\n")
            print(f"Normalization stats saved to: {normalization_stats_file}")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. If you want to add a new dataset, please add it to the list of MRI datasets or CT datasets.")

    # Create train and validation datasets
    if augmentation_type == "basic":
        train_dataset = Dataset(train_data, transform=BasicAugmentTransform(voxel_size, image_size, normalization_stats, developer_mode))
    elif augmentation_type == "trivial":
        raise NotImplementedError("Trivial augmentation is not implemented yet.")  # TODO @Diane: Integrate TrivialAugment
    elif augmentation_type == "groupaugment":
        raise NotImplementedError("Group augmentation is not implemented yet.")  # TODO @Diane: Implement + integrate GroupAugment
    else:
        raise ValueError(f"Invalid augmentation type: {augmentation_type}")

    # Create validation dataset only if validation data exists
    if valid_data:
        val_dataset = Dataset(valid_data, transform=EvaluationTransform(voxel_size, image_size, normalization_stats, developer_mode))
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
