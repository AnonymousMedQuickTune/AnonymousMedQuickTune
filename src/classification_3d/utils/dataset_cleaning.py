import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
import nibabel as nib
import shutil
import re
import pandas as pd
from tqdm import tqdm

from src.classification_3d.utils.dataset_info import analyze_dataset_statistics, save_statistics_to_file


def natural_key(string_):
    """
    Natural sorting for directory names.

    Args:
        string_ (str): String to sort

    Returns:
        list: List of strings and integers
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def find_valid_image_and_segmentation_files(base_path, data_point):
    """
    Find image and segmentation files with flexible naming.
    Supports both flat structure (e.g., lipo dataset) and nested structure (e.g., liver dataset).
    
    Args:
        base_path (str): Base path to the dataset
        data_point (str): Directory name containing the files
        
    Returns:
        tuple: (img_path, seg_path) or (None, None) if not found
    """
    data_point_path = os.path.join(base_path, data_point)
    
    # Common image file name patterns
    image_patterns = [
        "image.nii.gz",
        "img.nii.gz", 
        "image.nii",
        "img.nii",
        "image.nrrd",
        "img.nrrd",
        "image.mha",
        "img.mha"
    ]
    
    # Common segmentation file name patterns
    segmentation_patterns = [
        "segmentation.nii.gz",
        "seg.nii.gz",
        "mask.nii.gz",
        "label.nii.gz",
        "segmentation.nii",
        "seg.nii",
        "mask.nii",
        "label.nii",
        "segmentation.nrrd",
        "seg.nrrd",
        "mask.nrrd",
        "label.nrrd",
        "segmentation_lesion0.nii.gz",
        "segmentation_lesion1.nii.gz",
        "segmentation_lesion0_CNN.nii.gz",
        "segmentation_lesion1_CNN.nii.gz",
        "segmentation_lesion0_PhD.nii.gz",
        "segmentation_lesion1_PhD.nii.gz",
        "segmentation_lesion0_RAD.nii.gz",
        "segmentation_lesion1_RAD.nii.gz",
        "segmentation_lesion0_STUD1.nii.gz",
        "segmentation_lesion1_STUD1.nii.gz",
        "segmentation_lesion0_STUD2.nii.gz",
        "segmentation_lesion1_STUD2.nii.gz"
    ]
    
    def search_files_recursively(directory, patterns):
        """Search for files recursively in directory and subdirectories"""
        found_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                for pattern in patterns:
                    if file == pattern:
                        found_files.append(os.path.join(root, file))
        return found_files
    
    # Try to find image file (first check direct, then recursive)
    img_path = None
    
    # First try direct search (for flat structure like lipo)
    for pattern in image_patterns:
        test_path = os.path.join(data_point_path, pattern)
        if os.path.exists(test_path):
            img_path = test_path
            break
    
    # If not found, try recursive search (for nested structure like liver)
    if img_path is None:
        found_images = search_files_recursively(data_point_path, image_patterns)
        if found_images:
            img_path = found_images[0]  # Take first match
    
    # Try to find segmentation file (first check direct, then recursive)
    seg_path = None
    
    # First try direct search (for flat structure like lipo)
    for pattern in segmentation_patterns:
        test_path = os.path.join(data_point_path, pattern)
        if os.path.exists(test_path):
            seg_path = test_path
            break
    
    # If not found, try recursive search (for nested structure like liver)
    if seg_path is None:
        found_segmentations = search_files_recursively(data_point_path, segmentation_patterns)
        if found_segmentations:
            # Check if we have multiple lesion files that need to be combined
            lesion_files = [f for f in found_segmentations if 'segmentation_lesion' in os.path.basename(f)]
            if len(lesion_files) > 1:
                # Multiple lesion files found - we'll combine them in copy_and_convert_files
                # For now, return the first one as a marker that we need to combine
                seg_path = lesion_files[0]
            else:
                seg_path = found_segmentations[0]  # Take first match
    
    # Debug: Print what files are actually in the directory
    if not img_path or not seg_path:
        try:
            print(f"Debug: Searching in {data_point_path}")
            for root, dirs, files in os.walk(data_point_path):
                level = root.replace(data_point_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
        except Exception as e:
            print(f"Debug: Could not list files in {data_point_path}: {e}")
    
    return img_path, seg_path


def find_all_lesion_segmentation_files(base_path, data_point):
    """
    Find all lesion segmentation files (e.g., segmentation_lesion0.nii.gz, segmentation_lesion1.nii.gz).
    
    Args:
        base_path (str): Base path to the dataset
        data_point (str): Directory name containing the files
        
    Returns:
        list: List of paths to all lesion segmentation files, sorted
    """
    data_point_path = os.path.join(base_path, data_point)
    found_files = []
    
    for root, dirs, files in os.walk(data_point_path):
        for file in files:
            # Match patterns like segmentation_lesion0.nii.gz, segmentation_lesion1.nii.gz, etc.
            if file.startswith("segmentation_lesion") and (file.endswith(".nii.gz") or file.endswith(".nii")):
                found_files.append(os.path.join(root, file))
    
    # Sort to ensure consistent ordering (lesion0, lesion1, etc.)
    return sorted(found_files)


def combine_lesion_segmentations(lesion_file_paths, output_path, segmentation_type):
    """
    Combine multiple lesion segmentation files into a single segmentation file.
    All non-zero voxels from all lesion files are combined (union).
    
    Note: This combination method creates a binary mask where each voxel indicates
    whether there is a lesion (1) or not (0), but it does NOT preserve the distinction
    between different lesion instances. After combination:
    - Semantic segmentation is possible: we can determine which pixels contain lesions
    - Instance segmentation is NOT possible: we cannot distinguish between different
      lesion instances (e.g., lesion0 vs lesion1) as all lesions are merged into a
      single binary mask. If instance-level information is needed, the original
      separate lesion files should be preserved.
    
    Args:
        lesion_file_paths (list): List of paths to lesion segmentation files
        output_path (str): Path where the combined segmentation will be saved
        segmentation_type (str): Type of segmentation to create ("semantic" or "instance")
        
    Returns:
        bool: True if successful, False otherwise
    """
    if segmentation_type != "semantic":
        raise NotImplementedError(f"Please update the combine_lesion_segmentations function in dataset_cleaning.py to handle {segmentation_type} segmentation.")

    if not lesion_file_paths:
        return False
    
    try:
        # Load the first lesion file to get the reference shape and affine
        first_lesion = nib.load(lesion_file_paths[0])
        combined_array = first_lesion.get_fdata().astype(np.uint16)
        affine = first_lesion.affine
        
        # Combine all other lesion files
        for lesion_path in lesion_file_paths[1:]:
            lesion_img = nib.load(lesion_path)
            lesion_array = lesion_img.get_fdata().astype(np.uint16)
            
            # Ensure shapes match
            if lesion_array.shape != combined_array.shape:
                print(f"Warning: Shape mismatch in {lesion_path}: {lesion_array.shape} vs {combined_array.shape}")
                continue
            
            # Combine: take union (max) of all non-zero values
            # Note: This creates a binary mask (0 or 1) where 1 indicates any lesion.
            # This enables semantic segmentation (lesion vs. background) but NOT instance
            # segmentation (cannot distinguish between different lesion instances).
            combined_array = np.maximum(combined_array, lesion_array)
        
        # Create combined segmentation image
        combined_img = nib.Nifti1Image(combined_array.astype(np.uint16), affine)
        nib.save(combined_img, output_path)
        return True
        
    except Exception as e:
        print(f"Error combining lesion segmentations: {e}")
        return False


def get_valid_directories(original_path):
    """
    Get list of directories that have valid image and segmentation files.
    
    Args:
        original_path (str): Path to the original dataset directory
        
    Returns:
        list: List of valid directories
    """
    # Get all directories, excluding non-data directories
    directory_names = [data_point for data_point in sorted(os.listdir(original_path), key=natural_key) 
                      if os.path.isdir(os.path.join(original_path, data_point)) and data_point != "preprocessed"]
    
    # Check which directories have both image and segmentation files
    valid_directories = []
    for data_point in directory_names:
        img_path, seg_path = find_valid_image_and_segmentation_files(original_path, data_point)
        if img_path and seg_path:
            valid_directories.append(data_point)
        else:
            print(f"Warning: Skipping {data_point} - missing image or segmentation file")
    
    print(f"Found {len(valid_directories)} valid samples out of {len(directory_names)} total")
    return valid_directories


def copy_and_convert_files(original_path, cleaned_path, valid_directories, dataset_name, segmentation_type):
    """
    Copy and convert files from valid directories to cleaned dataset.
    
    This function:
    1. Renumbers directories sequentially (e.g., Lipo-001, Lipo-002, ...)
    2. Standardizes file formats to .nii.gz
    3. Uses consistent naming (image.nii.gz, segmentation.nii.gz)
    4. Handles format conversion with fallback to direct copy
    
    Args:
        original_path (str): Path to the original dataset directory
        cleaned_path (str): Path to the cleaned dataset directory
        valid_directories (list): List of valid directories (already filtered for missing files)
        dataset_name (str): Name of the dataset ('lipo', 'desmoid', 'gist')
        segmentation_type (str): Type of segmentation to create ("semantic" or "instance")
    """
    # Check if dataset size is supported (3-digit numbering limit)
    if len(valid_directories) > 999:
        raise NotImplementedError(f"Dataset {dataset_name} has more than 999 samples, which is not supported yet.")
    
    # Process each valid directory and renumber them sequentially
    print(f"Copying and converting {len(valid_directories)} files...")
    for new_idx, old_dir in tqdm(enumerate(valid_directories, start=1), 
                                  total=len(valid_directories), 
                                  desc="", 
                                  unit="file"):
        # Create new directory name with 3-digit padding (e.g., "Lipo-001", "Gist-073")
        new_dir = f"{dataset_name.capitalize()}-{new_idx:03d}"
        
        # Find the original image and segmentation files with flexible naming
        # Supports: image.nii.gz, img.nii, image.nrrd, segmentation.nii.gz, mask.nii, etc.
        old_img_path, old_seg_path = find_valid_image_and_segmentation_files(original_path, old_dir)
        
        # Create new directory structure in cleaned dataset
        new_dir_path = os.path.join(cleaned_path, new_dir)
        os.makedirs(new_dir_path, exist_ok=True)
        
        # Create standardized file paths (always image.nii.gz and segmentation.nii.gz)
        new_img_path = os.path.join(new_dir_path, "image.nii.gz")
        new_seg_path = os.path.join(new_dir_path, "segmentation.nii.gz")
        
        # Convert image file with fallback
        try:    
            # Try to load and save with nibabel for format conversion
            img = nib.load(old_img_path)
            nib.save(img, new_img_path)
        except Exception as e:
            # If conversion fails, copy the original file directly
            print(f"Warning: Could not convert image {old_img_path}: {e}")
            shutil.copy2(old_img_path, new_img_path)
        
        # Check if we have multiple lesion segmentation files that need to be combined
        lesion_seg_paths = find_all_lesion_segmentation_files(original_path, old_dir)
        
        if len(lesion_seg_paths) > 1:
            # Multiple lesion files found - combine them into one segmentation
            if not combine_lesion_segmentations(lesion_seg_paths, new_seg_path, segmentation_type):
                # Fallback: if combination fails, use the first lesion file
                print(f"Warning: Failed to combine lesion segmentations for {old_dir}, using first lesion file")
                try:
                    seg = nib.load(lesion_seg_paths[0])
                    nib.save(seg, new_seg_path)
                except Exception as e:
                    print(f"Warning: Could not convert segmentation {lesion_seg_paths[0]}: {e}")
                    shutil.copy2(lesion_seg_paths[0], new_seg_path)
        elif len(lesion_seg_paths) == 1:
            # Single lesion file - just convert it
            try:
                seg = nib.load(lesion_seg_paths[0])
                nib.save(seg, new_seg_path)
            except Exception as e:
                print(f"Warning: Could not convert segmentation {lesion_seg_paths[0]}: {e}")
                shutil.copy2(lesion_seg_paths[0], new_seg_path)
        else:
            # No lesion files found, use the standard segmentation file
            # Convert segmentation file with fallback
            try:
                # Try to load and save with nibabel for format conversion
                seg = nib.load(old_seg_path)
                nib.save(seg, new_seg_path)
            except Exception as e:
                # If conversion fails, copy the original file directly
                print(f"Warning: Could not convert segmentation {old_seg_path}: {e}")
                shutil.copy2(old_seg_path, new_seg_path)


def update_csv_file(original_path, cleaned_path, valid_directories, dataset_name):
    """
    Update CSV file to match valid directories.
    
    Args:
        original_path (str): Path to the original dataset directory
        cleaned_path (str): Path to the cleaned dataset directory
        valid_directories (list): List of valid directories
        dataset_name (str): Name of the dataset ('lipo', 'desmoid', 'gist', 'liver')
    """
    # Check if CSV file exists
    original_csv_path = os.path.join(original_path, f"{dataset_name}_labels.csv")
    if not os.path.exists(original_csv_path):
        raise FileNotFoundError(f"CSV file not found in {original_path}")
    
    # Read original CSV file
    df = pd.read_csv(original_csv_path)
    
    # Filter labels based on valid directories
    valid_labels = []
    for old_dir in valid_directories:
        try:
            # Extract directory number from name
            # Handle different naming conventions:
            # - "Lipo-001" -> 1
            # - "Liver-001_MR" -> 1
            # - "Gist-073" -> 73
            # Extract number after the first dash
            dir_num_str = old_dir.split('-')[1]
            # Remove any suffix after underscore (e.g., "_MR")
            if '_' in dir_num_str:
                dir_num_str = dir_num_str.split('_')[0]
            dir_num = int(dir_num_str)
            
            # Convert to 0-based index because we use 0-based indexing for the labels
            if dir_num - 1 < len(df):  # 1-1=0, 2-1=1, 3-1=2, ...
                # Append valid label to list
                valid_labels.append(df.iloc[dir_num - 1])
            else:
                print(f"Warning: Directory {old_dir} (number {dir_num}) exceeds CSV length ({len(df)})")
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not match directory {old_dir} to label: {e}")
            continue
    
    # Create new DataFrame with valid labels
    new_df = pd.DataFrame(valid_labels)
    new_csv_path = os.path.join(cleaned_path, f"{dataset_name}_labels.csv")

    # Save new CSV file
    new_df.to_csv(new_csv_path, index=False)
    
    print(f"Updated CSV: {len(new_df)} labels out of {len(df)} original")
    return new_df


def clean_dataset(data_path, dataset_name, segmentation_type):
    """
    Clean the dataset by removing samples with missing files and renumbering them.
    
    Args:
        data_path (str): Path to the original dataset
        dataset_name (str): Name of the dataset ('lipo', 'desmoid', 'gist')
        segmentation_type (str): Type of segmentation to create ("semantic" or "instance")

    Returns:
        str: Path to the cleaned dataset
    """
    print(f"\n{'='*60}")
    print(f"Cleaning {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Define paths
    original_path = os.path.join(data_path, dataset_name)
    cleaned_path = os.path.join(data_path, f"{dataset_name}_cleaned")
    
    # Create cleaned directory
    os.makedirs(cleaned_path, exist_ok=True)
    
    # Get valid directories
    print(f"\nStep 1: Validating directories...")
    valid_directories = get_valid_directories(original_path)
    
    # Copy and convert files
    print(f"\nStep 2: Copying and converting files...")
    copy_and_convert_files(original_path, cleaned_path, valid_directories, dataset_name, segmentation_type)
    
    # Update CSV file
    print(f"\nStep 3: Updating CSV file...")
    new_df = update_csv_file(original_path, cleaned_path, valid_directories, dataset_name)
    
    # Analyze dataset statistics
    print(f"\nStep 4: Analyzing dataset statistics...")
    statistics = analyze_dataset_statistics(cleaned_path, new_df)
    
    # Save statistics to file
    statistics_file = os.path.join(cleaned_path, "statistics.txt")
    additional_info = {
        "Total samples": len(valid_directories),
        "Original samples": len(valid_directories) + len([d for d in os.listdir(original_path) 
                                                        if os.path.isdir(os.path.join(original_path, d)) 
                                                        and d != "preprocessed"]) - len(valid_directories),
        "Removed samples": len([d for d in os.listdir(original_path) 
                              if os.path.isdir(os.path.join(original_path, d)) 
                              and d != "preprocessed"]) - len(valid_directories)
    }
    save_statistics_to_file(statistics, statistics_file, dataset_name, additional_info)
    
    print(f"\n{'='*60}")
    print(f"Cleaning completed successfully!")
    print(f"Statistics saved to: {statistics_file}")
    print(f"Cleaned dataset saved to: {cleaned_path}")
    print(f"{'='*60}\n")
    return cleaned_path