# general imports
import os

# libraries
from tqdm import tqdm
import nibabel as nib
import numpy as np

# functions
from .utils import load_image, resample_image, normalize_image, save_image, spacing_info, img_path
from .crop_pad import get_image_dimensions, crop_scan, extract_liver, tumor_bbox, pad_3d_image 

# global variables
image_name = "image.nii.gz"
segmentation_name = "segmentation.nii.gz"
# General and image paths - these will be set dynamically
path = None
path_img = None

# create new folder/temp folder
new_path = './gist_final'

def preprocessing(file_paths, new_path, voxel):
    # Loop through each file path and process images and segmentations
    print("---- Start -----")
    for count, file in enumerate(file_paths, start=1):

        img_file = os.path.join(file, image_name)
        seg_file = os.path.join(file, segmentation_name)
        image = load_image(img_file)
        segmentation = load_image(seg_file)
        
        resampled_img = resample_image(image, voxel)
        resampled_seg = resample_image(segmentation, voxel)
        
        norm_image = normalize_image(resampled_img)
        norm_seg = normalize_image(resampled_seg)
        
        # Extract the original directory name from the file path
        original_dir_name = os.path.basename(file)
        
        # Save the normalized images and segmentations
        img_path, seg_path = save_image(norm_image, norm_seg, count, new_path, original_dir_name)
        
        print(f"Image No {count} done")
    print("---- End -----")


def main_preprocessing(file_paths, new_path, voxel):
    # Resample and normalization of images
    # Creates the new images after resampling and normalization and saves them
    preprocessing(file_paths, new_path, voxel)

    # To check the difference between only resampling and normalization 
    # jupyter notebook is better

    # Path for image and segmentation after resampling and normalizing
    # get IDs of all images sizes < 50 and > 75% quartile
    img_files = os.listdir(new_path)
    img_IDs = [i[5:13] for i in img_files]
    fname_list = os.listdir(new_path)

    threshold = list() 
    threshold_ids = list()
    (x_75, y_75, z_75), (x_median, y_median, z_median) = get_image_dimensions(new_path)
    
    for mask_file in tqdm(fname_list):
        print(mask_file[5:13])
        # match the mask file to the original image
        if mask_file[5:13] in img_IDs:
            index = img_IDs.index(mask_file[5:13])
            seg = nib.load(new_path + '/' + mask_file + "/" + segmentation_name) # load segmentation
            
            seg_data = seg.get_fdata()
            sx, sy, sz = seg_data.shape
            if (sx > x_75) or (sy > y_75) or (sz > z_75) or (sx < 50) or (sy < 50) or (sz < 50):
                print(sx, sy, sz)
                threshold.append(new_path + '/' + mask_file + "/" + segmentation_name)
                threshold_ids.append(mask_file)

    # Processing pipeline over all images above the 75% quartile
    failed_list = []

    # Get median sizes and below 75% quartile size
    max_bbox_size, (medx, medy, medz) = get_image_dimensions(new_path)

    # Overrides created images after cropping and padding
    for mask_file in tqdm(fname_list):
        print(mask_file[5:13])
        # match the mask file to the original image
        #if mask_file[5:13] in threshold_ids[5:13]:
        if any(mask_file[5:13] == ids[5:13] for ids in threshold_ids):
            index = img_IDs.index(mask_file[5:13])
            img = nib.load(new_path + '/' + mask_file + "/" + image_name) # load image
            seg = nib.load(new_path + '/' + mask_file + "/" + segmentation_name) # load segmentation
            
            img_data = img.get_fdata()
            seg_data = seg.get_fdata()

            print("Original image shape: ", img_data.shape)
            # get the header information
            img_header = img.header 
            seg_header = seg.header 
            img_affine = img.affine
            seg_affine = seg.affine

            # select the tumor region
            mask_data =  extract_liver(seg_data, liver = False)
            if mask_data is None:
                print(f'[WARNING]: roi. {mask_file}')
                failed_list.append(mask_file)
                continue

            #if fixed_size_box:
            bbox = tumor_bbox(mask_data, max_bbox_size, bbox_size = [medx, medy, medz])
            print("new bbox: ", bbox)
            if bbox is None:
                print(f'[WARNING]: bbox is None. {mask_file}')
                failed_list.append(mask_file)
                continue
            min_row, min_col, min_slice, max_row, max_col, max_slice = bbox
            print(f"row min-max: {min_row,max_row}, col min-max: {min_col, max_col}, slice min-max: {min_slice, max_slice} ") # This is not giving the correct value back.

            # crop the scan
            scan_crop = crop_scan(img_data, bbox) 
            seg_crop = crop_scan(seg_data, bbox) 
            
            cropped_img = nib.Nifti1Image(scan_crop, img_affine, img_header)
            cropped_seg = nib.Nifti1Image(seg_crop, seg_affine, seg_header)
            
            new_img = cropped_img.get_fdata()
            new_seg = cropped_seg.get_fdata()

            print("New image shape: ", new_img.shape)

            # # Check that each dimension is at least 50 in one line
            dims_ok = all(dim >= 50 for dim in new_img.shape)

            if not dims_ok:
                # Adding padding if necessary
                padded_img_data = pad_3d_image(cropped_img)
                padded_seg_data = pad_3d_image(cropped_seg)

                # Create a new nibabel image object with the padded data
                final_img = nib.Nifti1Image(padded_img_data, img_affine, img_header)
                final_seg = nib.Nifti1Image(padded_seg_data, seg_affine, seg_header)
                
            else: 
                # Create a new nibabel image object with the padded data
                final_img = cropped_img
                final_seg = cropped_seg            
    
            temp_img_path = new_path + '/' + mask_file + "/" + image_name
            temp_seg_path = new_path + '/' + mask_file +  "/" + segmentation_name
            
            # Save the new NIfTI file
            
            print(f"saving to: {temp_img_path}, overriding image: {mask_file}")
            nib.save(final_img, temp_img_path)
            nib.save(final_seg, temp_seg_path)


    print(failed_list)



if __name__ == "__main__":
    main()