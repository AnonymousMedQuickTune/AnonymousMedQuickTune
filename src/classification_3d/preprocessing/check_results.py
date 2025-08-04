import os
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def show_vx_sz(image_name):
    # Initialize lists to store voxel sizes and image dimensions from two separate file paths
    voxel_sizes_1 = []
    image_sizes_1 = []
    voxel_sizes_2 = []
    image_sizes_2 = []

    # Loop through the file indices
    for i in range(1, 116):
        # Construct file paths using formatted strings
        file_path_1 = f"../WORCDatabase/Lipo/worc/Lipo-{i:03d}/" + image_name
        file_path_2 = f"./padding/Lipo-{i:03d}/" + image_name
        
        # Load the image files
        file_1 = nib.load(file_path_1)
        file_2 = nib.load(file_path_2)
        
        # Retrieve voxel sizes from the file headers
        voxel_sizes_1.append(file_1.header.get_zooms())
        voxel_sizes_2.append(file_2.header.get_zooms())
        
        # Get image data as a NumPy array and store its dimensions
        image_sizes_1.append(file_1.get_fdata().shape)
        image_sizes_2.append(file_2.get_fdata().shape)

    # Set up the figure for 12 subplots (2 rows x 6 columns)
    fig, axs = plt.subplots(3, 4, figsize=(30, 10))  # Adjust the figsize as needed

    # Titles for each subplot
    titles = [
        'Voxel x (File 1)', 'Size x (File 1)','Voxel x (File 2)', 'Size x (File 2)',
        'Voxel y (File 1)', 'Size y (File 1)','Voxel y (File 2)', 'Size y (File 2)',
        'Voxel z (File 1)', 'Size z (File 1)','Voxel z (File 2)', 'Size z (File 2)'
    ]

    # Prepare data for plotting
    data = [
        [v[0] for v in voxel_sizes_1],  [s[0] for s in image_sizes_1],[v[0] for v in voxel_sizes_2], [s[0] for s in image_sizes_2],
        [v[1] for v in voxel_sizes_1], [s[1] for s in image_sizes_1], [v[1] for v in voxel_sizes_2], [s[1] for s in image_sizes_2],
        [v[2] for v in voxel_sizes_1], [s[2] for s in image_sizes_1], [v[2] for v in voxel_sizes_2], [s[2] for s in image_sizes_2]
    ]

    # Plot each histogram
    for ax, d, title in zip(axs.flatten(), data, titles):
        ax.hist(d, bins=20, edgecolor='black', linewidth=1.2)
        ax.set_title(title)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def show_scan():
    # Paths to the images and masks
    image_paths = [
        '../WORCDatabase/Lipo/worc/Lipo-001/image.nii.gz',
        './padding/Lipo-001/image.nii.gz'
    ]
    mask_paths = [
        '../WORCDatabase/Lipo/worc/Lipo-001/segmentation.nii.gz',
        './padding/Lipo-001/segmentation.nii.gz'
    ]

    # Load the images and masks
    images = [nib.load(path).get_fdata() for path in image_paths]
    masks = [nib.load(path).get_fdata() for path in mask_paths]

    # Indices to display specific slices
    slice_indices = [2, 5]  # First for original, second for resampled

    # Titles for subplots
    titles = ['Original Image', 'Segmentation Mask', 'Resample Image', 'Segmentation + Resampled']

    # Set up the figure
    plt.figure(figsize=(20, 10))

    # Loop to display each image and corresponding mask
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        # Determine the image and slice index based on i
        img = images[i // 2]  # Integer division to alternate between images
        msk = masks[i // 2]
        slice_idx = slice_indices[i // 2]
        
        # Show image
        plt.imshow(img[:, :, slice_idx], cmap='gray')
        
        # If index is odd, overlay the mask
        if i % 2 == 1:
            plt.imshow(msk[:, :, slice_idx], alpha=0.5, cmap='viridis')
        
        # Add title
        plt.title(titles[i])

    # Show the plot
    plt.tight_layout()
    plt.show()

# Function to save plots
# Create a custom colormap with red and high transparency
colors = [(1, 0, 0, alpha) for alpha in np.linspace(0, 1, 256)]  # Red with varying transparency
red_translucent = LinearSegmentedColormap.from_list('RedTranslucent', colors, N=256)

# Create a custom colormap with blue and high transparency
colors_2 = [(0, 0, 1, alpha) for alpha in np.linspace(0, 1, 256)]  # Blue with varying transparency
blue_translucent = LinearSegmentedColormap.from_list('BlueTranslucent', colors_2, N=256)


def save_plot(image, mask, image2, mask2, slice_idx, filename):
    plt.figure(figsize=(16, 4))
    
    # Plot the oroginal image
    plt.subplot(1, 4, 1)
    plt.imshow(image[:, :, slice_idx], cmap='gray')
    plt.title('Original Image')

    # Plot the original image with the segmentation mask
    plt.subplot(1, 4, 2)
    plt.imshow(image[:, :, slice_idx], cmap='gray')
    plt.contour(mask[:, :, slice_idx], levels=[0.5], cmap=blue_translucent)
    plt.title('Segmentation Mask')

    # Plot the processed image
    plt.subplot(1, 4, 3)
    plt.imshow(image2[:, :, slice_idx], cmap='gray')
    plt.title('Processed Image')

    # Plot the processed image with the segmentation mask
    plt.subplot(1, 4, 4)
    plt.imshow(image2[:, :, slice_idx], cmap='gray')
    plt.contour(mask2[:, :, slice_idx], levels=[0.5], cmap=blue_translucent)
    plt.title('Processed Mask')

    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def show_scans():
    # Paths to the images and masks
    # Original images and masks
    image_paths = [f'../WORCDatabase/GIST/worc/GIST-{i:03d}/image.nii.gz' for i in range(1, 116)]
    mask_paths = [f'../WORCDatabase/GIST/worc/GIST-{i:03d}/segmentation.nii.gz' for i in range(1, 116)]
    # New images + masks
    image_paths2 = [f'../WORCDatabase/GIST/worc/GIST-{i:03d}/image.nii.gz' for i in range(1, 116)]
    mask_paths2 = [f'../WORCDatabase/GIST/worc/GIST-{i:03d}/segmentation.nii.gz' for i in range(1, 116)]
    
    # Create a directory to save the plots
    os.makedirs('plots', exist_ok=True)

    # Loop through all images and masks
    for i, (image_path, mask_path, img_path, seg_path) in enumerate(zip(image_paths, mask_paths, image_paths2, mask_paths2)):
        # Load the images and masks
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        image2 = nib.load(img_path).get_fdata()
        mask2 = nib.load(seg_path).get_fdata()
        
        
        # Choose a slice index (you can adjust this as needed)
        slice_idx = image2.shape[2] // 2  # Using the middle slice as an example
        
        # Save the plot
        save_plot(image, mask, image2, mask2, slice_idx, f'C:/Users/ovied/Desktop/BigR/XAI/plots/plot_{i:03d}.png')

    # Create an HTML file to display all the plots
    html_content = '<html><body>'
    for i in range(115):
        html_content += f'<h3>Image {i+1}</h3>'
        html_content += f'<img src="C:/Users/ovied/Desktop/BigR/XAI/plots/plot_{i:03d}.png" style="width:800px;"><br><br>'
    html_content += '</body></html>'

    # Write the HTML content to a file
    with open('./scans/gist_scans.html', 'w') as f:
        f.write(html_content)

    print("HTML file created successfully: index.html")


def main():
    image_name = "image.nii.gz"
    #show_vx_sz(image_name)
    show_scans()



if __name__ == "__main__":
    main()