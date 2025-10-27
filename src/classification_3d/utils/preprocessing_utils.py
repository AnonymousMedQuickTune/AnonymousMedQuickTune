import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from src.classification_3d.utils.dataset_cleaning import find_valid_image_and_segmentation_files, natural_key

from src.classification_3d.preprocessing.crop_pad import extract_liver, tumor_bbox, crop_scan, pad_3d_image

import SimpleITK as sitk

def get_paths(dataset_path, dataset_name):
    """
    Get paths to images, segmentations, and CSV file for a given dataset path.
    
    Args:
        dataset_path (str): Path to the dataset directory
        dataset_name (str, optional): Name of the dataset for flexible CSV file detection
        
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
    
    csv_path = os.path.join(dataset_path, f"{dataset_name}_labels.csv")

    return images_path, segmentations_path, csv_path


# TODO @Natalia: Pls double check this implementation + compare calculated voxel size with values you worked with so far
# NOTE: Pls see experimental_setting.yaml > data.voxel_calculation
# NOTE: Pls see cleaned_dataset_path/preprocessed_*/statistics.txt
# TODO @Diane: Check if voxel size is also in (W, H, D) format and not (H, W, D) format
def calculate_voxel_size_from_images(cleaned_dataset_path, dataset_name, calculation_method="median"):
    """
    Calculate voxel for a dataset using the specified calculation method.
    
    Args:
        cleaned_dataset_path (str): Path to the cleaned dataset
        dataset_name (str, optional): Name of the dataset for flexible CSV file detection
        calculation_method (str): Method to calculate voxel size:
            - 'mean': Calculate mean voxel size across all training images
            - 'median': Calculate median voxel size across all training images
            - 'isotropic': Return (1.0, 1.0, 1.0)
            - 'volumetric_isotropic': Calculate isotropic voxel based on median volume
    
    Returns:
        tuple: Voxel size as (x, y, z) tuple
    """
    # Get image paths for the dataset
    images_path, _, _ = get_paths(cleaned_dataset_path, dataset_name)

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

# ------------------------------------------------------------------------------------------------
# Resampling and normalization (NNU-Net approach for MRI datasets)
# ------------------------------------------------------------------------------------------------
def get_image_dimensions_from_input(file_paths, image_file_name):
    """
    Get image dimensions from input files before processing.
    
    Args:
        file_paths (list): List of directory paths containing images
        image_file_name (str): Name of the image file (e.g., "image.nii.gz")
        
    Returns:
        tuple: ((x_75, y_75, z_75), (x_median, y_median, z_median))
    """
    # Build image paths from input file paths
    image_paths = [os.path.join(file_path, image_file_name) for file_path in file_paths]
    
    # Initialize lists to store sizes
    x_sizes, y_sizes, z_sizes = [], [], []

    # Retrieve the x, y, z sizes for each image
    for img_path in image_paths:
        if os.path.exists(img_path):
            image = nib.load(img_path)
            sx, sy, sz = image.header.get_data_shape()[:3]
            x_sizes.append(sx)
            y_sizes.append(sy)
            z_sizes.append(sz)

    # Check if we have any images
    if len(x_sizes) == 0:
        raise ValueError("No images found to calculate dimensions from input files")

    # Compute the 75% quartile for each dimension
    x_75 = np.percentile(x_sizes, 75)
    y_75 = np.percentile(y_sizes, 75)
    z_75 = np.percentile(z_sizes, 75)

    # Compute the median for each dimension
    x_median = np.median(x_sizes)
    y_median = np.median(y_sizes)
    z_median = np.median(z_sizes)

    return (x_75, y_75, z_75), (x_median, y_median, z_median)

def resample_image(image, voxel_size, interpolator=sitk.sitkLinear, default_value=None):
    """
    Resample a SimpleITK.Image to the target voxel size.

    Args:
        image (SimpleITK.Image): Input image (axes are logical (x, y, z) = (W, H, D))
        voxel_size (tuple[float,float,float]): Target voxel size in (x, y, z) order [mm]
        interpolator: sitk.sitkLinear for intensity images, sitk.sitkNearestNeighbor for labels
        default_value (optional): Fill value for areas outside the original FOV (default is None)
    Returns:
        SimpleITK.Image
    """
    # Validate inputs
    voxel_size = tuple(float(v) for v in voxel_size)
    assert len(voxel_size) == image.GetDimension(), "voxel_size must match image dimension"

    # Get original spacing and size
    original_spacing = np.array(image.GetSpacing(), dtype=float)
    original_size = np.array(image.GetSize(), dtype=int)
    
    # Compute new image size to preserve FOV (Field of View) when changing voxel spacing.
    # NOTE @Natalia:
    # Use np.round() instead of np.ceil() to avoid artificially enlarging the FOV.
    # Added np.maximum(1, ...) to prevent zero-sized dimensions due to rounding errors (caused by np.round()).
    new_size = np.maximum(1, np.round(original_size * (original_spacing / np.array(voxel_size))).astype(int))


    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(voxel_size)  # (x, y, z) order, float values required by SimpleITK
    resampler.SetSize([int(x) for x in new_size])  # Convert to int list for SimpleITK compatibility
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # Output pixel type
    if interpolator == sitk.sitkNearestNeighbor:
        # Keep label/integer type for segmentations
        resampler.SetOutputPixelType(image.GetPixelID())
        if default_value is None:
            default_value = 0
    else:
        # Use float for intensities to avoid truncation
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    
    if default_value is not None:
        resampler.SetDefaultPixelValue(default_value)
    
    # Return resampled image
    return resampler.Execute(image)


def should_use_masked_normalization(original_size_xyz, cropped_size_xyz, threshold: float = 0.25) -> bool:
    """
    Decide whether to switch to masked normalization based on cropping reduction,
    as in nnU-Net: if average patient size drops by >= 1/4, use masked normalization.
    Sizes are (x,y,z) tuples from SimpleITK GetSize().
    
    Args:
        original_size_xyz (tuple): Original size in (x, y, z) format
        cropped_size_xyz (tuple): Cropped size in (x, y, z) format
        threshold (float): Threshold for cropping reduction (default is 0.25 according to nnU-Net)

    Returns:
        bool: Whether to use masked normalization
    """
    orig_vox = int(np.prod(original_size_xyz))
    crop_vox = int(np.prod(cropped_size_xyz))
    if orig_vox == 0:
        return False
    reduction = (orig_vox - crop_vox) / float(orig_vox)
    return reduction >= threshold

def normalize_mri_image_nnunet(image: sitk.Image, use_masked: bool) -> sitk.Image:
    """
    Apply resampling and MRI normalization to the images following the nnU-Net approach:
    - If use_masked: z-score within non-zero voxels (post-crop), set outside to 0.
    - Else: simple per-patient z-score over entire volume.

    Args:
        image (SimpleITK.Image): Input image (axes are logical (x, y, z) = (W, H, D))
        use_masked (bool): Whether to use masked normalization

    Returns:
        SimpleITK.Image
    """
    # Get image array
    image_array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)  # (z,y,x)

    if use_masked:
        # Calculate mean and standard deviation from non-zero voxels only
        mask = image_array != 0
        if np.any(mask):
            mean = float(image_array[mask].mean())
            std = float(image_array[mask].std())
        else:
            # Fallback: if all voxels are zero, use original values
            mean = float(image_array.mean())
            std = float(image_array.std())
        
        # Avoid division by zero
        if std == 0.0:
            std = 1e-6

        # Z-score normalization
        normalized_image_array = (image_array - mean) / std
        normalized_image_array[~mask] = 0.0
    else:
        # Calculate mean and standard deviation from entire volume
        mean = float(image_array.mean())
        std = float(image_array.std())

        # Avoid division by zero
        if std == 0.0:
            std = 1e-6
        
        # Z-score normalization
        normalized_image_array = (image_array - mean) / std

    # Create a new SimpleITK image with the normalized array
    normalized_image = sitk.GetImageFromArray(normalized_image_array)
    normalized_image.CopyInformation(image)  # Copy metadata

    # Cast to float32 to save ~50% of disk space by normalization
    return sitk.Cast(normalized_image, sitk.sitkFloat32)

def crop_and_pad_tumor_region(image, segmentation, x_75, y_75, z_75, x_median, y_median, z_median):
    """
    Crop and pad the tumor region of the image and segmentation.
    
    Args:
        image (sitk.Image): The image to crop and pad.
        segmentation (sitk.Image): The segmentation to crop and pad.
        x_75, y_75, z_75: 75th percentile dimensions
        x_median, y_median, z_median: Median dimensions
        
    Returns:
        tuple: (cropped_image, cropped_segmentation) as SimpleITK images
    """
    # Convert SimpleITK to numpy arrays
    # SimpleITK uses (x, y, z) indexing but GetArrayFromImage returns (z, y, x)
    img_array = sitk.GetArrayFromImage(image)  # Shape: (z, y, x) or (depth, height, width)
    seg_array = sitk.GetArrayFromImage(segmentation)  # Shape: (z, y, x)
    
    # Get image metadata
    origin = image.GetOrigin()
    spacing = image.GetSpacing()  # This is (x, y, z) spacing
    direction = image.GetDirection()
    
    # Extract tumor region (ROI)
    mask_data = extract_liver(seg_array, liver=False)
    if mask_data is None:
        print('[WARNING]: No ROI found in segmentation')
        return image, segmentation
    
    # Calculate bounding box using the statistics passed in
    # The bbox will be returned in (min_row, min_col, min_slice, max_row, max_col, max_slice) = (z, y, x) format
    # but we need to pass dimensions in the order that the bbox represents
    # max_bbox_size and bbox_size should be in (z, y, x) order for consistency with the (z, y, x) array
    max_bbox_size = [z_75, y_75, x_75]  # Reorder from (x, y, z) to (z, y, x) to match array format
    bbox_size = [z_median, y_median, x_median]  # Reorder from (x, y, z) to (z, y, x)
    
    try:
        bbox = tumor_bbox(mask_data, max_bbox_size, bbox_size=bbox_size)
        
        # Crop the scan and segmentation using the bounding box
        # crop_scan expects (min_row, min_col, min_slice, ...) which corresponds to (z, y, x) for the array
        cropped_img_data = crop_scan(img_array, bbox)
        cropped_seg_data = crop_scan(seg_array, bbox)
        
        # Check if dimensions are acceptable (minimum 50 voxels per dimension)
        dims_ok = all(dim >= 50 for dim in cropped_img_data.shape)
        
        if not dims_ok:
            # Need to create temporary nibabel images for pad_3d_image
            # pad_3d_image expects nibabel image objects
            temp_img = nib.Nifti1Image(cropped_img_data, np.eye(4))
            temp_seg = nib.Nifti1Image(cropped_seg_data, np.eye(4))
            
            # Apply padding
            final_img_data = pad_3d_image(temp_img)
            final_seg_data = pad_3d_image(temp_seg)
        else:
            final_img_data = cropped_img_data
            final_seg_data = cropped_seg_data
        
        # Convert back to SimpleITK images
        # sitk.GetImageFromArray expects (z, y, x) numpy array and converts to (x, y, z) SimpleITK
        final_img = sitk.GetImageFromArray(final_img_data)
        final_seg = sitk.GetImageFromArray(final_seg_data)
        
        # Preserve spatial information
        # spacing is in (x, y, z) format, which matches SimpleITK's orientation
        final_img.SetSpacing(spacing)
        final_seg.SetSpacing(spacing)
        final_img.SetOrigin(origin)
        final_seg.SetOrigin(origin)
        final_img.SetDirection(direction)
        final_seg.SetDirection(direction)
        
        return final_img, final_seg
        
    except Exception as e:
        print(f'[WARNING]: Bbox calculation or cropping failed: {str(e)}')
        return image, segmentation


def resize_gist_images(image, segmentation):
    """
    Resize GIST images to a fixed size to reduce memory usage.
    
    This function resizes both image and segmentation to (96, 96, 96) voxels
    to prevent out-of-memory errors with larger GIST volumes.
    
    Args:
        image (sitk.Image): The image to resize
        segmentation (sitk.Image): The segmentation to resize
        
    Returns:
        tuple: (resized_image, resized_segmentation) as SimpleITK images
    """
    # Convert SimpleITK to numpy arrays
    # SimpleITK uses (x, y, z) indexing but GetArrayFromImage returns (z, y, x)
    img_array = sitk.GetArrayFromImage(image)  # Shape: (z, y, x)
    seg_array = sitk.GetArrayFromImage(segmentation)  # Shape: (z, y, x)
    
    # Get image metadata
    origin = image.GetOrigin()
    spacing = image.GetSpacing()  # This is (x, y, z) spacing
    direction = image.GetDirection()
    
    # Fixed size for GIST to prevent OOM
    target_size = (96, 96, 96)
    
    # Calculate zoom factors
    current_shape = img_array.shape
    zoom_factors = [target_size[i] / current_shape[i] for i in range(3)]
    
    # Resize image and segmentation
    resized_img_data = zoom(img_array, zoom_factors, order=1)  # Linear interpolation
    resized_seg_data = zoom(seg_array, zoom_factors, order=0)  # Nearest neighbor
    
    print(f"GIST: Resized from {current_shape} to {resized_img_data.shape}")
    
    # Convert back to SimpleITK images
    # sitk.GetImageFromArray expects (z, y, x) numpy array and converts to (x, y, z) SimpleITK
    final_img = sitk.GetImageFromArray(resized_img_data)
    final_seg = sitk.GetImageFromArray(resized_seg_data)
    
    # Preserve spatial information
    # spacing is in (x, y, z) format, which matches SimpleITK's orientation
    final_img.SetSpacing(spacing)
    final_seg.SetSpacing(spacing)
    final_img.SetOrigin(origin)
    final_seg.SetOrigin(origin)
    final_img.SetDirection(direction)
    final_seg.SetDirection(direction)
    
    return final_img, final_seg


def save_preprocessed_images_and_segmentations_to_nifti(image, image_file_name, segmentation, segmentation_file_name, out_root, patient_id):
    """
    Save preprocessed image and segmentation to NIfTI (.nii.gz) format.

    This function:
      - Casts the image to float32 to reduce memory footprint.
      - Ensures the segmentation is stored as integer labels.
      - Checks for matching geometry (size) between image and segmentation.
      - Writes both files with compression enabled.

    Args:
        image (sitk.Image): The preprocessed medical image (e.g., MRI or CT volume).
        image_file_name (str): Name of the image file (e.g., "image.nii.gz").
        segmentation (sitk.Image): The corresponding segmentation mask.
        segmentation_file_name (str): Name of the segmentation file (e.g., "segmentation.nii.gz").
        out_root (str): Root output directory for saving processed cases.
        patient_id (str): Identifier for the current patient/case (used as subfolder name).

    Raises:
        ValueError: If the image and segmentation sizes do not match.

    Returns:
        tuple[str, str]: Paths to the saved image and segmentation files.
    """
    # Create output directory for the current case
    case_dir = os.path.join(out_root, patient_id)
    os.makedirs(case_dir, exist_ok=True)

    # Cast image to float32 (~50% smaller than float64)
    if image.GetPixelID() != sitk.sitkFloat32:
        image = sitk.Cast(image, sitk.sitkFloat32)

    # Replace NaN/Inf values (which can corrupt NIfTI files)
    arr = sitk.GetArrayFromImage(image)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, copy=False)
        image = sitk.GetImageFromArray(arr)
        image.CopyInformation(segmentation)

    # Ensure segmentation is integer type
    if segmentation.GetPixelID() in (sitk.sitkFloat32, sitk.sitkFloat64):
        seg_arr = np.rint(sitk.GetArrayFromImage(segmentation)).astype(np.uint16, copy=False)
        segmentation = sitk.GetImageFromArray(seg_arr)
        segmentation.CopyInformation(image)

    # Validate geometry consistency (same voxel grid)
    if image.GetSize() != segmentation.GetSize():
        raise ValueError(f"Image and segmentation sizes differ: {image.GetSize()} vs {segmentation.GetSize()}")

    # Build file paths
    img_path = os.path.join(case_dir, image_file_name)
    seg_path = os.path.join(case_dir, segmentation_file_name)

    # Save both files with compression
    sitk.WriteImage(image, img_path, useCompression=True)
    sitk.WriteImage(segmentation, seg_path, useCompression=True)
