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
        "label.nrrd"
    ]
    
    # Try to find image file
    img_path = None
    for pattern in image_patterns:
        test_path = os.path.join(data_point_path, pattern)
        if os.path.exists(test_path):
            img_path = test_path
            break
    
    # Try to find segmentation file
    seg_path = None
    for pattern in segmentation_patterns:
        test_path = os.path.join(data_point_path, pattern)
        if os.path.exists(test_path):
            seg_path = test_path
            break
    
    return img_path, seg_path


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


def copy_and_convert_files(original_path, cleaned_path, valid_directories, dataset_name):
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
        
        # Convert segmentation file with fallback
        try:
            # Try to load and save with nibabel for format conversion
            seg = nib.load(old_seg_path)
            nib.save(seg, new_seg_path)
        except Exception as e:
            # If conversion fails, copy the original file directly
            print(f"Warning: Could not convert segmentation {old_seg_path}: {e}")
            shutil.copy2(old_seg_path, new_seg_path)


def update_csv_file(original_path, cleaned_path, valid_directories):
    """
    Update CSV file to match valid directories.
    
    Args:
        original_path (str): Path to the original dataset directory
        cleaned_path (str): Path to the cleaned dataset directory
        valid_directories (list): List of valid directories
    """
    # Check if CSV file exists
    original_csv_path = os.path.join(original_path, "dataset.csv")
    if not os.path.exists(original_csv_path):
        raise FileNotFoundError(f"CSV file not found in {original_path}")
    
    # Read original CSV file
    df = pd.read_csv(original_csv_path)
    
    # Filter labels based on valid directories
    valid_labels = []
    for old_dir in valid_directories:
        try:
            # Extract directory number from name (e.g. "Lipo-001" -> 1)
            dir_num = int(old_dir.split('-')[-1])  # 1, 2, 3, ...
            # Convert to 0-based index because we use 0-based indexing for the labels
            if dir_num - 1 < len(df):  # 1-1=0, 2-1=1, 3-1=2, ...
                # Append valid label to list
                valid_labels.append(df.iloc[dir_num - 1])
        except (ValueError, IndexError):
            print(f"Warning: Could not match directory {old_dir} to label")
            continue
    
    # Create new DataFrame with valid labels
    new_df = pd.DataFrame(valid_labels)
    new_csv_path = os.path.join(cleaned_path, "dataset.csv")

    # Save new CSV file
    new_df.to_csv(new_csv_path, index=False)
    
    print(f"Updated CSV: {len(new_df)} labels out of {len(df)} original")
    return new_df


def clean_dataset(data_path, dataset_name):
    """
    Clean the dataset by removing samples with missing files and renumbering them.
    
    Args:
        data_path (str): Path to the original dataset
        dataset_name (str): Name of the dataset ('lipo', 'desmoid', 'gist')
        
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
    copy_and_convert_files(original_path, cleaned_path, valid_directories, dataset_name)
    
    # Update CSV file
    print(f"\nStep 3: Updating CSV file...")
    new_df = update_csv_file(original_path, cleaned_path, valid_directories)
    
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