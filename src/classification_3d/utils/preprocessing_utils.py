import os
import math
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import skimage

from src.classification_3d.utils.dataset_cleaning import find_valid_image_and_segmentation_files, natural_key


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

# NOTE: Pls see experimental_setting.yaml > data.voxel_calculation
# NOTE: Pls see cleaned_dataset_path/preprocessed_*/statistics.txt
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

def extract_largest_component(mask):
    """
    Extract the largest connected component from a binary mask.
    
    Args:
        mask: Binary mask (can contain multiple disconnected regions)
        
    Returns:
        Binary mask containing only the largest connected component, or None if no component found
    """
    mask = mask.astype(int)
    
    # Check if mask contains multi-class labels (e.g., 0=background, 1=liver, 2=tumor)
    unique_values = np.unique(mask)
    has_label_2 = 2 in unique_values
    
    if has_label_2:
        raise NotImplementedError("Multi-class mask detected. This is not supported yet.")
        
    # Find connected components in the mask (separate regions get different labels: 1, 2, 3, ...)
    # connectivity=2 means 2D connectivity (only within slices, not between slices)
    # Note: Labels are assigned in order of discovery (typically top-left to bottom-right),
    # so label 1 is the FIRST found component, not necessarily the largest
    labeled = skimage.measure.label(mask, connectivity=2)

    # Find the largest connected component by comparing sizes
    # Get properties of all regions to find which one is largest
    props = skimage.measure.regionprops(labeled)
    if len(props) == 0:
        mask_largest_component = labeled
    else:
        # Find the label of the largest component (by area/number of pixels)
        largest_label = max(props, key=lambda x: x.area).label
        
        # Keep only the largest connected component, remove all others
        labeled[labeled != largest_label] = 0
        mask_largest_component = labeled
    
    # Check if any component was found (if mask is all zeros, no component exists)
    if np.count_nonzero(mask_largest_component) == 0:
        print('[WARNING]: no connected component found in mask')
        return None
    return mask_largest_component

def get_center_of_mass_3D(binary_image):
    """
    Calculate the center of mass (centroid) of a binary 3D image.
    
    This function finds all connected components in the binary image and calculates
    the center of mass. If multiple components exist, it returns the mean of all
    centroids. For a single component (as expected when used with extract_largest_component),
    this gives the centroid of that component.
    
    Args:
        binary_image: Binary 3D numpy array (0 = background, 1 = foreground)
        
    Returns:
        tuple: Center of mass coordinates as (row, col, slice) = (z, y, x) format
        
    Note:
        The function uses the mean of all component centroids if multiple components exist.
        For best results, use with a mask containing only a single connected component.
    """
    # Label the connected components in the binary image
    # Each separate region gets a different label (1, 2, 3, ...)
    labeled_image = skimage.measure.label(binary_image)
    
    # Compute region properties including center of mass (centroid) for all components
    props = skimage.measure.regionprops_table(labeled_image, properties=['centroid'])
    
    # Get the center of mass coordinates
    # If multiple components exist, this calculates the mean of all centroids
    # For a single component, this is simply the centroid of that component
    center_row = np.mean(props['centroid-0'])
    center_col = np.mean(props['centroid-1'])
    center_slice = np.mean(props['centroid-2'])
    
    # Return the center of mass coordinates as a tuple (z, y, x)
    center_of_mass = (center_row, center_col, center_slice)
    return center_of_mass
    
def tumor_bbox(mask_region_of_interest, max_bbox_size, bbox_size=None):
    """
    Calculate bounding box around the region of interest based on center of mass.
    If multiple components are present, the center is set to the mean value of all centroids of all components
    
    This function:
    1. Finds the center of mass of the region of interest mask
    2. Gets the initial bounding box from the mask
    3. Adjusts the bbox size based on the actual region of interest size and max constraints
    4. Returns a bounding box centered around the center of mass
    
    Args:
        mask_region_of_interest: Binary mask of the region of interest (numpy array)
        max_bbox_size: Maximum allowed bbox size for each dimension [z, y, x]
        bbox_size: Desired bbox size for each dimension [z, y, x] (default: [36, 36, 36])
        
    Returns:
        tuple: Bounding box coordinates (min_row, min_col, min_slice, max_row, max_col, max_slice)
               Coordinates are in (z, y, x) format
    """
    # Fix mutable default argument: create a copy if None is provided
    if bbox_size is None:
        bbox_size = [36, 36, 36]
    else:
        # Create a copy to avoid modifying the original list
        bbox_size = list(bbox_size)
    
    # Step 1: Calculate the center of mass of the largest connected component mask
    # This gives us the center point around which we'll build the bounding box
    com = get_center_of_mass_3D(mask_region_of_interest)
    com = np.array(com).astype(int)

    # Step 2: Get the initial bounding box from the mask properties
    # This gives us the actual extent of the region of interest (min/max coordinates)
    # If multiple components exist, we need to find the bounding box that encompasses ALL components
    image_probs = skimage.measure.regionprops(mask_region_of_interest)
    
    # Check if any region was found
    if len(image_probs) == 0:
        raise ValueError("No connected component found in mask_region_of_interest")
    
    # Calculate bounding box that encompasses ALL components
    # For multiple components, we take the min/max across all components
    all_bboxes = [props.bbox for props in image_probs]
    # bbox format: (min_row, min_col, min_slice, max_row, max_col, max_slice)
    orig_min_row = min(bbox[0] for bbox in all_bboxes)
    orig_min_col = min(bbox[1] for bbox in all_bboxes)
    orig_min_slice = min(bbox[2] for bbox in all_bboxes)
    orig_max_row = max(bbox[3] for bbox in all_bboxes)
    orig_max_col = max(bbox[4] for bbox in all_bboxes)
    orig_max_slice = max(bbox[5] for bbox in all_bboxes)

    # Step 3: Adjust bbox_size based on actual region of interest size
    # If the region of interest (all components combined) is larger than desired bbox_size, use max_bbox_size instead
    # This ensures we capture the entire region of interest while respecting size limits
    actual_size_z = orig_max_row - orig_min_row
    actual_size_y = orig_max_col - orig_min_col
    actual_size_x = orig_max_slice - orig_min_slice
    
    bbox_size[0] = bbox_size[0] if actual_size_z < bbox_size[0] else max_bbox_size[0]
    bbox_size[1] = bbox_size[1] if actual_size_y < bbox_size[1] else max_bbox_size[1]
    bbox_size[2] = bbox_size[2] if actual_size_x < bbox_size[2] else max_bbox_size[2]

    # Step 4: Calculate final bounding box centered around the center of mass
    # The bbox is centered at the center of mass of the region of interest with the adjusted size
    # Note: We need to get the image shape to ensure bbox doesn't exceed image boundaries
    # This will be handled by the caller, but we calculate it here for safety
    min_row = int(max(0, com[0] - bbox_size[0] // 2))
    max_row = int(com[0] + bbox_size[0] // 2)
    min_col = int(max(0, com[1] - bbox_size[1] // 2))
    max_col = int(com[1] + bbox_size[1] // 2)
    min_slice = int(max(0, com[2] - bbox_size[2] // 2))
    max_slice = int(com[2] + bbox_size[2] // 2)
    
    return min_row, min_col, min_slice, max_row, max_col, max_slice

def crop_scan(scan, bbox):
    """
    Crop a 3D scan (image or segmentation) to the specified bounding box.
    
    Args:
        scan: Original 3D scan as numpy array (shape: z, y, x)
        bbox: Bounding box coordinates as tuple (min_row, min_col, min_slice, max_row, max_col, max_slice)
              Coordinates are in (z, y, x) format
        
    Returns:
        numpy array: Cropped scan with shape determined by the bounding box
    """
    min_row, min_col, min_slice, max_row, max_col, max_slice = bbox

    # Crop the scan using the bounding box coordinates
    # Array slicing: scan[z_min:z_max, y_min:y_max, x_min:x_max]
    scan_crop = scan[min_row:max_row, min_col:max_col, min_slice:max_slice]
    return scan_crop

def pad_3d_image(image):
    """
    Pad a 3D image to ensure minimum dimensions of 36 voxels per axis.
    
    This function pads each dimension that is smaller than 36 voxels to reach
    the minimum size, while preserving the original content. Padding is applied
    symmetrically (equal amounts on both sides).
    
    Args:
        image: Nibabel image object (3D) to pad
        
    Returns:
        numpy array: Padded image data with minimum dimensions of 36x36x36
    """
    # Get the current shape of the image
    current_shape = image.shape
    # Calculate target shape: ensure each dimension is at least 36 voxels
    target_shape = tuple(max(dim, 36) for dim in current_shape)
    
    # Extract image data as float32 array
    img_data = image.get_fdata().astype(np.float32)
    # Calculate the padding amounts for each dimension (symmetric padding)
    # Each dimension gets padded equally on both sides to reach target_shape
    pad_depth = math.ceil((target_shape[0] - current_shape[0]) / 2)
    pad_height = math.ceil((target_shape[1] - current_shape[1]) / 2)
    pad_width = math.ceil((target_shape[2] - current_shape[2]) / 2)    
    
    # Calculate final shape after padding
    final_shape = [current_shape[0] + pad_depth * 2, current_shape[1] + pad_height * 2, current_shape[2] + pad_width * 2]
    
    # Adjust padding if final shape exceeds target (due to rounding)
    # Reduce padding by 1 on one side to match target_shape exactly
    pad_depth_conditioned = pad_depth - 1 if final_shape[0] > target_shape[0] else pad_depth
    pad_height_conditioned = pad_height - 1 if final_shape[1] > target_shape[1] else pad_height
    pad_width_conditioned = pad_width - 1 if final_shape[2] > target_shape[2] else pad_width

    # Apply symmetric padding using np.pad
    # Padding format: ((before_z, after_z), (before_y, after_y), (before_x, after_x))
    # mode='constant' pads with zeros
    padded_image = np.pad(img_data, ((pad_depth, pad_depth_conditioned), (pad_height, pad_height_conditioned), (pad_width, pad_width_conditioned)), mode='constant')
    
    return padded_image

def crop_and_pad_tumor_region(image, segmentation, x_75, y_75, z_75, x_median, y_median, z_median, model_task):
    """
    Crop and pad the tumor region of the image and segmentation.
    
    Args:
        image (sitk.Image): The image to crop and pad.
        segmentation (sitk.Image): The segmentation to crop and pad.
        x_75, y_75, z_75: 75th percentile dimensions
        x_median, y_median, z_median: Median dimensions
        model_task (str): Type of machine learning task: classification, semantic_segmentation, instance_segmentation

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
    
    # Extract the largest connected component from the segmentation mask
    # NOTE @Natalia:
    # - Updated this from extract_liver 
    # - The largest component is now extracted instead of the first one (bug solved)
    # - Liver dataset has no extra class for the liver itself, so we don't need to handle it here.
    if model_task == "classification":
        mask_region_of_interest = extract_largest_component(seg_array)
        if mask_region_of_interest is None:
            print('[WARNING]: No ROI found in segmentation')
            return image, segmentation
    else:
        raise NotImplementedError("Semantic and instance segmentation is not supported yet. Think about if it makes sense to extract the largest component for your segmentation study. For instance segmentation it might be an issue or if the total volume of all tumors is of interest.")
        # NOTE: in tumor_bbox the center of mass is used to calculate the bounding box.
        # if you want to keep multiple components that center is set to the mean value of all centroids of all components.

    # Calculate bounding box using the statistics passed in
    # The bbox will be returned in (min_row, min_col, min_slice, max_row, max_col, max_slice) = (z, y, x) format
    # but we need to pass dimensions in the order that the bbox represents
    # max_bbox_size and bbox_size should be in (z, y, x) order for consistency with the (z, y, x) array
    max_bbox_size = [z_75, y_75, x_75]  # Reorder from (x, y, z) to (z, y, x) to match array format
    bbox_size = [z_median, y_median, x_median]  # Reorder from (x, y, z) to (z, y, x)
    
    try:
        # NOTE: If multiple components are present, the center of the tumor bbox is set to the mean value of all centroids of all components
        bbox = tumor_bbox(mask_region_of_interest, max_bbox_size, bbox_size=bbox_size)
        
        # Crop the scan and segmentation using the bounding box
        # crop_scan expects (min_row, min_col, min_slice, ...) which corresponds to (z, y, x) for the array
        cropped_img_data = crop_scan(img_array, bbox)
        cropped_seg_data = crop_scan(seg_array, bbox)
        
        # Check if dimensions are acceptable (minimum 36 voxels per dimension)
        dims_ok = all(dim >= 36 for dim in cropped_img_data.shape)
        
        if not dims_ok:
            # Need to create temporary nibabel images for pad_3d_image
            # pad_3d_image expects nibabel image objects
            temp_img = nib.Nifti1Image(cropped_img_data, np.eye(4))
            temp_seg = nib.Nifti1Image(cropped_seg_data, np.eye(4))
            
            # Apply padding
            # NOTE @Natalia:
            # - Fixed bug in padding the width
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


def resize_gist_melanoma_images(image, segmentation):
    """
    Resize GIST and MELANOMA images to reduce memory usage.
    
    This function reduces only the depth (Z) dimension to a maximum of 96 voxels,
    while preserving the width (X) and height (Y) dimensions. This preserves
    X-Y resolution (important for diagnostic features) while reducing computational cost.
    
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
    
    # Get SimpleITK size for consistent printing (x, y, z format)
    sitk_size_before = image.GetSize()  # (x, y, z)
    
    # Calculate target size: only reduce depth (Z) dimension to max 96
    # Note: img_array.shape is (z, y, x), so:
    #   - Index 0 = z (depth) - reduce to max 96
    #   - Index 1 = y (height) - keep unchanged
    #   - Index 2 = x (width) - keep unchanged
    current_shape_np = img_array.shape  # (z, y, x)
    target_size_np = (
        min(current_shape_np[0], 96),  # z (depth) - reduce to max 96
        current_shape_np[1],            # y (height) - unchanged
        current_shape_np[2]             # x (width) - unchanged
    )
    
    # Check if resizing is needed
    if current_shape_np == target_size_np:
        # No resizing needed, return original images
        return image, segmentation
    
    # Calculate zoom factors (1.0 means no change for that dimension)
    zoom_factors = [target_size_np[i] / current_shape_np[i] for i in range(3)]
    
    # Resize image and segmentation
    resized_img_data = zoom(img_array, zoom_factors, order=1)  # Linear interpolation
    resized_seg_data = zoom(seg_array, zoom_factors, order=0)  # Nearest neighbor
    
    # Convert back to SimpleITK to get final size in (x, y, z) format for consistent printing
    temp_img = sitk.GetImageFromArray(resized_img_data)
    sitk_size_after = temp_img.GetSize()  # (x, y, z)
    
    # Print in consistent (x, y, z) format to match other print statements
    print(f"GIST/Melanoma: Resized depth from (x, y, z) = {sitk_size_before} to (x, y, z) = {sitk_size_after}")
    
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


def resize_lipo_desmoid_liver_images(image, segmentation):
    """
    Resize LIPO, DESMOID, and LIVER images to reduce memory usage.
    
    This function resizes:
    - Height (Y) and Width (X) dimensions to 256 voxels
    - Depth (Z) dimension to 32 voxels
    
    This reduces computational cost while maintaining reasonable spatial resolution.
    
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
    
    # Get SimpleITK size for consistent printing (x, y, z format)
    sitk_size_before = image.GetSize()  # (x, y, z)
    
    # Calculate target size:
    # Note: img_array.shape is (z, y, x), so:
    #   - Index 0 = z (depth) - resize to 32
    #   - Index 1 = y (height) - resize to 256
    #   - Index 2 = x (width) - resize to 256
    current_shape_np = img_array.shape  # (z, y, x)
    target_size_np = (
        32,   # z (depth) - resize to 32
        256,  # y (height) - resize to 256
        256   # x (width) - resize to 256
    )
    
    # Check if resizing is needed
    if current_shape_np == target_size_np:
        # No resizing needed, return original images
        return image, segmentation
    
    # Calculate zoom factors (1.0 means no change for that dimension)
    zoom_factors = [target_size_np[i] / current_shape_np[i] for i in range(3)]
    
    # Resize image and segmentation
    resized_img_data = zoom(img_array, zoom_factors, order=1)  # Linear interpolation
    resized_seg_data = zoom(seg_array, zoom_factors, order=0)  # Nearest neighbor
    
    # Convert back to SimpleITK to get final size in (x, y, z) format for consistent printing
    temp_img = sitk.GetImageFromArray(resized_img_data)
    sitk_size_after = temp_img.GetSize()  # (x, y, z)
    
    # Print in consistent (x, y, z) format to match other print statements
    print(f"LIPO/Desmoid/Liver: Resized from (x, y, z) = {sitk_size_before} to (x, y, z) = {sitk_size_after}")
    
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
