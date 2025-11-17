import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import pickle
from tqdm import tqdm
import SimpleITK as sitk

from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    RandRotated,
    RandZoomd,
    RandFlipd,
    ResizeWithPadOrCropd,
    EnsureTyped,
)
from monai.utils import InterpolateMode
from monai.data import Dataset
from torch.utils.data import DataLoader

from src.classification_3d.utils.normalization_stats import calculate_normalization_stats
from src.classification_3d.utils.dataset_cleaning import clean_dataset
from src.classification_3d.utils.dataset_info import analyze_dataset_statistics, save_statistics_to_file, save_cv_split_info
from src.classification_3d.utils.preprocessing_utils import (
    get_paths,
    calculate_voxel_size_from_images,
    crop_and_pad_tumor_region,
    should_use_masked_normalization,
    normalize_mri_image_nnunet,
    save_preprocessed_images_and_segmentations_to_nifti,
    get_image_dimensions_from_input,
    resample_image,
    resize_worcdatabase_images,
    resize_depth_dimension
)
import datetime

def smart_preprocessing(file_paths, output_path, voxel_size, is_mri, dataset_name, model_task):
    """
    Comprehensive preprocessing pipeline that includes resampling, normalization, 
    empty slice removal, and tumor region cropping.
    
    Args:
        file_paths (list): List of file paths to the images and segmentations
        output_path (str): Path to the output directory where processed images will be saved
        voxel_size (tuple): Target voxel size in (x, y, z) format
        is_mri (bool): Whether the dataset is MRI or CT
        dataset_name (str): Name of the dataset for dataset-specific preprocessing
        model_task (str): Type of machine learning task: classification, semantic_segmentation, instance_segmentation

    Returns:
        dict: Dictionary containing preprocessing statistics and metadata
    """
    # Set variables
    # NOTE: Update for datasets besides the WORCDatabase
    image_file_name = "image.nii.gz"
    segmentation_file_name = "segmentation.nii.gz"
    minimum_size = 36  # Minimum size for the image and segmentation to be considered valid

    # Calculate dimension statistics from input files (median and 75th percentile)
    print("Calculating dimension statistics from input files...")
    (x_75, y_75, z_75), (x_median, y_median, z_median) = get_image_dimensions_from_input(file_paths, image_file_name)
    print(f"dimension statistics: {x_75}, {y_75}, {z_75}, {x_median}, {y_median}, {z_median}")

    # Process each image/segmentation pair with progress tracking.
    for file in tqdm(file_paths, desc="", unit="image"):
        # Load image and segmentation from file path
        img_file = os.path.join(file, image_file_name)
        seg_file = os.path.join(file, segmentation_file_name)
            
        # Check if files exist
        if not os.path.exists(img_file) or not os.path.exists(seg_file):
            print(f"Warning: Image or segmentation not found at {file} under image name {image_file_name} or segmentation name {segmentation_file_name}")
            continue
        
        # Load NIfTI image and segmentation as SimpleITK Image objects
        # (SimpleITK uses spatial axes (x, y, z) which correspond to (W, H, D) in the MONAI pipeline)
        image = sitk.ReadImage(img_file)
        segmentation = sitk.ReadImage(seg_file)

        # Standardize orientation to RAS (Right-Anterior-Superior) for consistency
        # This ensures all volumes have the same orientation regardless of their original orientation
        # RAS is the standard orientation for medical imaging and deep learning pipelines
        image = sitk.DICOMOrient(image, "RAS")
        segmentation = sitk.DICOMOrient(segmentation, "RAS")

        # Resample image to target voxel size
        # NOTE @Natalia: Updated this from resample_image_old to resample_image with correct interpolation mode.
        print("Resampling image to target voxel size...")
        image = resample_image(image, voxel_size, interpolator=sitk.sitkLinear)
        segmentation = resample_image(segmentation, voxel_size, interpolator=sitk.sitkNearestNeighbor)

        # Print image format and metadata (for debugging)
        # print(f"\nAfter resample_image():")
        # print(f"  Image: {file}")
        # print(f"  Format: SimpleITK Image")
        # print(f"  Size (WxHxD): {image.GetSize()}")
        # print(f"  Spacing (voxel size in mm): {image.GetSpacing()}")
        # print(f"  Origin: {image.GetOrigin()}")
        # print(f"  Pixel Type: {image.GetPixelIDTypeAsString()}")

        # Crop/pad tumor region of the image and segmentation if needed
        # NOTE @Natalia:
        # - Updated this whole part from Step 2 and Step 3 and integrated it here before normalization is applied!
        print("Crop/pad image if needed...")
        size_before_cropping = image.GetSize()  # (x, y, z) format
        size_x, size_y, size_z = image.GetSize()
        if (size_x > x_75 or size_y > y_75 or size_z > z_75 or size_x < minimum_size or size_y < minimum_size or size_z < minimum_size):
            print(f"Image size (x, y, z) = ({size_x}, {size_y}, {size_z}) outside acceptable range, cropping/padding...")
            image, segmentation = crop_and_pad_tumor_region(image, segmentation, x_75, y_75, z_75, x_median, y_median, z_median, model_task)
        size_after_cropping = image.GetSize()  # (x, y, z) format
        print(f"Image size after cropping/padding (x, y, z) = {size_after_cropping}")

        # Resize worcdatabase images to reduce memory usage and speed up the preprocessing pipeline
        print("Resizing WORC database images to reduce memory usage and speed up the preprocessing pipeline...")
        image, segmentation = resize_worcdatabase_images(image, segmentation, dataset_name)

        # Only normalize if is_mri is True
        # NOTE @Natalia:
        # - No normalization for segmentation masks!
        # - Cast to back to float32 to save ~50% of disk space by normalization
        # - For CT datasets, normalization is done in the run pipeline based on training data statistics
        #   depending on the cross-validation folds following the nnU-Net approach.
        # - Calculate mean and standard deviation from non-zero voxels only
        # - Avoid division by zero: If standard deviation is 0, set it to 1e-6.
        if is_mri:
            print("Normalizing MRI images according to nnU-Net's approach...")
            use_masked = should_use_masked_normalization(size_before_cropping, size_after_cropping)
            image = normalize_mri_image_nnunet(image, use_masked)
        else:
            print("CT image normalization is done in the run pipeline based on training data statistics depending on the cross-validation folds according to nnU-Net's approach...")

        # Extract the original directory name from the file path
        # and save the processed images and segmentations
        # NOTE @Natalia:
        # - Preprocessed data got saved to some hardcoded directory (new_path = './gist_final') and not used as assumed
        original_dir_name = os.path.basename(file)
        save_preprocessed_images_and_segmentations_to_nifti(image, image_file_name, segmentation, segmentation_file_name, output_path, original_dir_name)
    
    print("Preprocessing completed successfully!")


def apply_smart_preprocessing(cleaned_dataset_path, voxel_size, calculation_method, is_mri, dataset_name, model_task):
    """
    Apply Natalia's smart preprocessing pipeline to the dataset.
    
    Args:
        cleaned_dataset_path (str): Path to the cleaned dataset
        voxel_size (tuple): Voxel size in (x, y, z) format
        calculation_method (str): Method to calculate voxel size ('mean', 'median', 'isotropic', 'volumetric_isotropic')
        is_mri (bool): Whether the dataset is MRI > MRI datasets need normalization in the preprocessing for each image individually
        dataset_name (str): Name of the dataset for flexible CSV file detection
        model_task (str): Type of machine learning task: classification, semantic_segmentation, instance_segmentation
        
    Returns:
        str: Path to the preprocessed dataset
    """
    print(f"\nApplying smart preprocessing to '{cleaned_dataset_path}'...\n")

    # Get image paths for the cleaned dataset
    images_path, segmentations_path, csv_path = get_paths(cleaned_dataset_path, dataset_name)
    
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
    
    # Determine dataset name for dataset-specific preprocessing
    dataset_name = os.path.basename(cleaned_dataset_path).replace('_cleaned', '')
    
    # Run the preprocessing pipeline from Natalia's preprocessing code base
    smart_preprocessing(file_paths, output_path, voxel_size, is_mri, dataset_name, model_task)

    # Analyze preprocessed dataset statistics
    print("\n=== Preprocessed Dataset Statistics Analysis ===")
    
    # Load the CSV file to get labels
    csv_path = os.path.join(cleaned_dataset_path, f"{dataset_name}_labels.csv")
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


def load_3d_dataset_with_outer_cv_splits(experiment_base_dir, dataset_name, data_path="datasets", seed=42, use_smart_preprocessing=True, voxel_calculation="median", cv_outer_fold=0, mode="train", cv_outer_folds_repeats=5, cv_outer_folds_splits=3, model_task="classification"):
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
        model_task (str): Type of machine learning task: classification, semantic_segmentation, instance_segmentation

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
        if os.path.exists(cleaned_dataset_path) and os.path.exists(os.path.join(cleaned_dataset_path, f"{dataset_name}_labels.csv")):  
            print(f"> Found existing cleaned dataset at {cleaned_dataset_path}, skipping dataset cleaning...\n")
        else:
            print("\nX Cleaned dataset not found, running dataset cleaning...\n")
            cleaned_dataset_path = clean_dataset(data_path, dataset_name, model_task)

        # Check if preprocessed dataset with the given voxel calculation method exists
        preprocessed_dataset_path = os.path.join(cleaned_dataset_path, f"preprocessed_{voxel_calculation}")
        if os.path.exists(preprocessed_dataset_path):
            print(f"> Found existing preprocessed dataset at {preprocessed_dataset_path}, skipping preprocessing...\n")
            # Get voxel size from existing cleaned data (we'll calculate it again)
            voxel_size = calculate_voxel_size_from_images(cleaned_dataset_path, dataset_name, calculation_method=voxel_calculation)
        else:
            print("X Preprocessed dataset not found, running preprocessing...\n")
            voxel_size = calculate_voxel_size_from_images(cleaned_dataset_path, dataset_name, calculation_method=voxel_calculation)
            preprocessed_dataset_path, voxel_size = apply_smart_preprocessing(cleaned_dataset_path, voxel_size, calculation_method=voxel_calculation, is_mri=is_mri, dataset_name=dataset_name, model_task=model_task)
        # Keep the CSV path from the cleaned directory
        csv_path = os.path.join(cleaned_dataset_path, f"{dataset_name}_labels.csv")
    
    else:
        raise NotImplementedError("Smart preprocessing must be applied to use this function.")

    # Get image and segmentation paths from preprocessed data
    images, segmentations, _ = get_paths(preprocessed_dataset_path, dataset_name)

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
    

def DataTransform(normalization_stats, developer_mode, spatial_size=None, is_training=True):
    """
    Transform the training, validation, and test data. For training set, it applies data augmentation.
    
    Args:
        normalization_stats (dict): Normalization statistics for CT images only; default None for MRI images
        developer_mode (bool): If True, uses smaller model target shape for faster development; default False
        spatial_size (tuple): Image size in (H, W, D) format for e.g., ViT; default None
        is_training (bool): If True, applies data augmentation; if False, no data augmentation; default True
    """
    # Base transform that is always applied
    transforms = [
        # Load NIfTI images; usually in (H, W, D) format
        LoadImaged(keys="image"),  
        # Debug: Print shape after LoadImaged
        # lambda data: print(f"DEBUG LoadImaged: {data['image'].shape}") or data,
        #
        # Ensure channels are first (for MONAI compatibility):
        # (H, W, D) -> (C, H, W, D) or (H, W, D, C) -> (C, H, W, D) depending on the dataset.
        EnsureChannelFirstd(keys="image"),
        # Debug: Print shape after EnsureChannelFirstd
        # lambda data: print(f"DEBUG EnsureChannelFirstd: {data['image'].shape}") or data,
        #
        # NOTE @Natalia:
        # Removed Spacingd because resampling to a target voxel size is already performed during preprocessing to ensure consistent spacing.
    ]

    # Add data normalization for CT images only.
    # > For CT: Data normalization statistics are either
    #   - calculated from the training data following nnU-Net's approach.
    #   - provided by NePS (AutoNorm).
    # > For MRI: Data normalization is done in the preprocessing per image/patient individually.
    is_mri = normalization_stats is None
    if not is_mri:
        transforms.append(
            NormalizeIntensityd(keys="image", subtrahend=normalization_stats["mean"][0], divisor=normalization_stats["std"][0])
        )

    # Add resizing if needed:
    # > For developer mode, we need to resize the image to a smaller size for faster development. Spatial size in the format (H, W, D).
    # > For some models e.g., ViT, we need to resize the image to a specific size. Spatial size in the format (H, W, D).
    needs_resizing = developer_mode or spatial_size is not None
    if needs_resizing:
        transforms.append(
            ResizeWithPadOrCropd(keys="image", spatial_size=spatial_size, mode="constant", constant_values=0)
        )
    
    # Add data augmentation for training set only; no data augmentation for validation and test set!
    if is_training:
        transforms.extend([
            # Flip the image along a random spatial axis (H, W, or D).
            # Note: spatial_axis refers to spatial dims only — not the channel dim (C in (C, H, W, D)).
            # NOTE @Natalia:
            # Updated this from spatial_axis=0 to spatial_axis=[0, 1, 2] to flip the image along a random spatial axis.
            RandFlipd(keys="image", prob=0.2, spatial_axis=[0, 1, 2]),
            #
            # Randomly rotate the image around the depth (z) axis by ±25°.
            # IMPORTANT: ranges are in radians, not degrees1
            # NOTE @Natalia:
            # Changed from (-25, 25) to np.deg2rad(25) for correct units.
            # Use trilinear interpolation for smooth intensity transitions (continuous medical images),
            # fill empty regions created by rotation using border values (to avoid black edges),
            # and keep the original spatial size after transformation for consistent batching.
            RandRotated(
                keys="image",
                range_x=0.0,
                range_y=0.0,
                range_z=np.deg2rad(25),  # 25 degrees about z
                prob=0.2,
                mode=InterpolateMode.TRILINEAR,  # NOTE: For segmentation tasks, use InterpolateMode.NEAREST for the mask.
                padding_mode="border",
                keep_size=True,
            ),
            # Randomly zoom the image by a factor between 0.8 and 1.2.
            # NOTE @Natalia:
            # Use trilinear interpolation for smooth intensity transitions (continuous medical images),
            # fill empty regions created by zooming using edge values (to avoid black edges),
            # and keep the original spatial size after transformation for consistent batching.
            RandZoomd(
                keys="image",
                prob=0.2,
                min_zoom=0.8,
                max_zoom=1.2,
                mode=InterpolateMode.TRILINEAR,  # NOTE: For segmentation tasks, use InterpolateMode.NEAREST for the mask.
                padding_mode="edge",
                keep_size=True,
            ),
        ])

    # Ensures the image is a torch.Tensor instead of a NumPy array.
    # Keeps MONAI's meta_dict synchronized, so spatial information (affine, spacing, etc.) remains attached and consistent throughout the pipeline.
    transforms.append(EnsureTyped(keys="image"))
    
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
    spatial_size=None,
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
        spatial_size (tuple): Spatial size in (H, W, D) format for ViT; default None
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
        train_dataset = Dataset(train_data, transform=DataTransform(normalization_stats, developer_mode, spatial_size=spatial_size, is_training=True))
    else:
        raise ValueError(f"Invalid augmentation type: {augmentation_type}")

    # Create validation dataset only if validation data exists
    if valid_data:
        val_dataset = Dataset(valid_data, transform=DataTransform(normalization_stats, developer_mode, spatial_size=spatial_size, is_training=False))
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
