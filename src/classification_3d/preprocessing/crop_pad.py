import os
import skimage
import nibabel as nib
import numpy as np
import math

def get_image_dimensions(data_path):
    # Get the list of image file paths
    image_name = "image.nii.gz"
    image_paths = [os.path.join(data_path, f, image_name) for f in os.listdir(data_path)]
    # Initialize lists to store sizes
    x_sizes, y_sizes, z_sizes = [], [], []

    # Retrieve the x, y, z sizes for each image
    for img_path in image_paths:
        image = nib.load(img_path)
        sx, sy, sz = image.header.get_data_shape()[:3]
        x_sizes.append(sx)
        y_sizes.append(sy)
        z_sizes.append(sz)

    # Compute the 75% quartile for each dimension
    x_75 = np.percentile(x_sizes, 75)
    y_75 = np.percentile(y_sizes, 75)
    z_75 = np.percentile(z_sizes, 75)

    # Compute the median for each dimension
    x_median = np.median(x_sizes)
    y_median = np.median(y_sizes)
    z_median = np.median(z_sizes)

    return (x_75, y_75, z_75), (x_median, y_median, z_median)

def extract_liver(mask, liver):
    """
    get either the liver or tumor region from the mask
    args:
        mask: mask of the liver and tumor
        liver: if True return the liver region, if False return the tumor region
    """
    mask = mask.astype(int)
    
    # Not necessary for tummor segmention for Lipo
    #if liver:
    #    mask[mask == 2] = 1
    #else:
    #    mask[mask == 1] = 0
    #    mask[mask == 2] = 1
    #    mask = mask.astype(int)
        
    labeled = skimage.measure.label(mask, connectivity=2)

    labeled[labeled != 1] = 0
    mask = labeled
    if np.count_nonzero(mask) == 0:
        print('[WARNING]: no liver found')
        return None
    return mask

def get_center_of_mass_3D(binary_image):
    # Label the connected components in the binary image
    labeled_image = skimage.measure.label(binary_image)
    
    # Compute region properties including center of mass
    props = skimage.measure.regionprops_table(labeled_image, properties=['centroid'])
    print(props)
    
    # Get the center of mass coordinates
    center_row = np.mean(props['centroid-0'])
    center_col = np.mean(props['centroid-1'])
    center_slice = np.mean(props['centroid-2'])
    
    # Return the center of mass coordinates as a tuple
    center_of_mass = (center_row, center_col, center_slice)
    return center_of_mass

def tumor_bbox(tumor_mask, max_bbox_size, bbox_size = [50, 50, 50]):
    '''
    find the center of mass of the tumor and return the bounding box
    args:
        tumor_mask: binary mask of the tumor
    returns:
        bbox: bounding box of the tumor (min_row, min_col, min_slice, max_row, max_col, max_slice = bbox)
    '''
    # find the center of mass of the tumor
    com = get_center_of_mass_3D(tumor_mask)
    com = np.array(com).astype(int)

    # get bbox
    image_probs = skimage.measure.regionprops((tumor_mask))
    for props in image_probs:
        bbox = props.bbox
        min_row, min_col, min_slice, max_row, max_col, max_slice = bbox 

    print("Median bbox size before change: ", bbox_size)
    print("Max size of a bbox should be: ", max_bbox_size)

    # Update bbox_size[0] based on comparison with bbox_size[0]
    bbox_size[0] = bbox_size[0] if (max_row - min_row) < bbox_size[0] else max_bbox_size[0]
    # Update bbox_size[1] based on comparison with bbox_size[1]
    bbox_size[1] = bbox_size[1] if (max_col - min_col) < bbox_size[1] else max_bbox_size[1]
    # Update bbox_size[2] based on comparison with bbox_size[2]
    bbox_size[2] = bbox_size[2] if (max_slice - min_slice) < bbox_size[2] else max_bbox_size[2]

    print("bbox: ", bbox_size)

    # extract bbox around the center of mass
    min_row = int(max(0, com[0] - bbox_size[0] // 2))
    max_row = int(com[0] + bbox_size[0]//2)
    min_col = int(max(0, com[1] - bbox_size[1] // 2))
    max_col = int(com[1] + bbox_size[1]//2)
    min_slice = int(max(0, com[2] - bbox_size[2] // 2))
    max_slice = int(com[2] + bbox_size[2]//2)
    
    return min_row, min_col, min_slice, max_row, max_col, max_slice


def crop_scan(scan, bbox):
    '''
    Function to crop the scan to the bounding box of the liver
    args:
        scan: original scan (3D) as numpy array
        bbox: bounding box of the liver (min_row, min_col, min_slice, max_row, max_col, max_slice = bbox)
        margin_frac: fraction of the image side to add to the bounding box

    returns:
        scan_crop: cropped scan
    '''
    min_row, min_col, min_slice, max_row, max_col, max_slice = bbox

    # Crop the scan using the updated bounding box
    scan_crop = scan[min_row:max_row, min_col:max_col, min_slice:max_slice]
    # scan_crop = scan[min_row:max_row, min_col:max_col, :]
    print("Scan crop shape: ", scan_crop.shape)
    return scan_crop


def pad_3d_image(image):
    """
    Pad a 3D image to the target shape while preserving the content.
    
    Args:
    - image: 3D numpy array representing the image.
    - target_shape: Tuple specifying the target shape (depth, height, width) of the padded image.
    
    Returns:
    - Padded 3D numpy array.
    """
    # Get the current shape of the image
    current_shape = image.shape
    target_shape = tuple(max(dim, 50) for dim in current_shape)
    
    img_data = image.get_fdata().astype(np.float32)
    print("current img shape", current_shape)
    print("target img shape", target_shape)
    # Calculate the padding amounts for each dimension
    pad_depth = math.ceil((target_shape[0] - current_shape[0]) / 2)
    pad_height = math.ceil((target_shape[1] - current_shape[1]) / 2)
    pad_width = math.ceil((target_shape[2] - current_shape[2]) / 2)    
    
    print(f"pad depth: {pad_depth}, pad height: {pad_height}, pad width: {pad_width} ") 
    
    print("Calculating if correct size of image is done to: ", target_shape)
    print(f"size x total: {current_shape[0] + pad_depth * 2}, size y total: {current_shape[1] + pad_height * 2 }, size z total: {current_shape[2] + pad_width * 2} ") 

    final_shape = [current_shape[0] + pad_depth * 2, current_shape[1] + pad_height * 2 , current_shape[2] + pad_width * 2]
    print("Final shape: ", final_shape)
    
    pad_depth_conditioned = pad_depth - 1 if final_shape[0] > target_shape[0] else pad_depth
    pad_height_conditioned = pad_height - 1 if final_shape[1] > target_shape[1] else pad_height
    pad_width_conditioned = pad_width - 1 if final_shape[2] > target_shape[2] else pad_width

    # Pad the image using np.pad
    padded_image = np.pad(img_data, ((pad_depth, pad_depth_conditioned), (pad_height, pad_height_conditioned), (pad_width_conditioned, pad_width)), mode='constant')
    
    return padded_image