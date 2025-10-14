
import os
import re
from typing import List
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
from statistics import median, mean
import numpy as np

# global variables
image_name = "/image.nii.gz"
segmentation_name = "/segmentation.nii.gz"

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# Function to calculate mean and median voxel dimensions
def calc_vx_dim(dims):
    mean_dim = {k: mean(dims[k]) for k in dims.keys()}
    median_dim = {k: median(dims[k]) for k in dims.keys()}
    return mean_dim, median_dim

# Spacing information
def spacing_info(path: List[str]):
    vx_dim = {'voxelx': [], 'voxely': [], 'voxelz': []}
    
    for file in path:
        image_file = nib.load(file)
        sx, sy, sz = image_file.header.get_zooms()
        vx_dim['voxelx'].append(sx)
        vx_dim['voxely'].append(sy)
        vx_dim['voxelz'].append(sz)
    
    mean_vx, median_vx = calc_vx_dim(vx_dim)
    
    # Convert the mean and median values to lists of floats
    mean_list = [float(mean_vx['voxelx']), float(mean_vx['voxely']), float(mean_vx['voxelz'])]
    median_list = [float(median_vx['voxelx']), float(median_vx['voxely']), float(median_vx['voxelz'])]
    return mean_list, median_list

# Get image paths
def img_path(type=None, data_path=None):
    # Define the base data path - use provided path or default
    if data_path is None:
        data_path = os.sep.join(["..","WORCDatabase","GIST","worc"])
    
    # Check if path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Get the list of directory names
    directory_names = os.listdir(data_path)
 
    if type == "img":
        # If type is Image, append image_name to directory names
        images = [os.path.join(data_path, f, image_name) for f in directory_names]
    elif type == "seg":
        # If type is Seg, append segmentation_name to directory names
        images = [os.path.join(data_path, f, segmentation_name) for f in directory_names]
    else:
        # Default case: return the directories
        images = [os.path.join(data_path, f) for f in directory_names]
    
    return images

# Resample, normalice and save new img
def load_image(file_path):
    image = sitk.ReadImage(file_path)
    return image

def resample_image(image, new_spacing):

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    # Ensure new_spacing is a list for SimpleITK compatibility
    if isinstance(new_spacing, tuple):
        new_spacing = list(new_spacing)
    elif isinstance(new_spacing, np.ndarray):
        new_spacing = new_spacing.tolist()
    
    # Force conversion to list if still not a list
    if not isinstance(new_spacing, list):
        new_spacing = list(new_spacing)

    new_size = original_size * (original_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(int)
    interpolator = sitk.sitkLinear
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size.tolist())
    # Convert to list of floats for SimpleITK compatibility
    spacing_list = [float(x) for x in new_spacing]
    resampler.SetOutputSpacing(spacing_list)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)

    resampled_image = resampler.Execute(image)
    return resampled_image

def normalize_image_old(image):
    # Get image array
    image_array = sitk.GetArrayFromImage(image)

    # Normalize to [0, 1]
    normalized_image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

    # Create a new SimpleITK image with the normalized array
    normalized_image = sitk.GetImageFromArray(normalized_image_array)
    normalized_image.CopyInformation(image)  # Copy metadata

    return normalized_image

def normalize_image(image):
    """
    Apply Z-score normalization to individual patient/image
    """
    # Get image array
    image_array = sitk.GetArrayFromImage(image)

    # Calculate mean and standard deviation
    mean = np.mean(image_array)
    std = np.std(image_array)

    # Avoid division by zero
    if std == 0:
        std = 1e-6
    
    # Z-score normalization
    normalized_image_array = (image_array - mean) / std

    # Create a new SimpleITK image with the normalized array
    normalized_image = sitk.GetImageFromArray(normalized_image_array)
    normalized_image.CopyInformation(image)  # Copy metadata

    return normalized_image

def save_image(images, seg, file_id, path, original_dir_name):
    # Use the original directory name
    new_dir_name = original_dir_name
    # Define the new directory path
    new_dir = Path(path) / new_dir_name
    # Create the new directory if it doesn't exist
    new_dir.mkdir(parents=True, exist_ok=True)
    # Define the file paths for the images and segmentation
    temp_img_path = Path(new_dir) / "image.nii.gz"
    temp_seg_path = new_dir / "segmentation.nii.gz"
    # Write the images and segmentation to the file paths
    sitk.WriteImage(images, temp_img_path)
    sitk.WriteImage(seg, temp_seg_path)
    
    return temp_img_path, temp_seg_path