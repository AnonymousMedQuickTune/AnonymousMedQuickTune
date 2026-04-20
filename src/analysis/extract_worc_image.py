#!/usr/bin/env python3
"""
Script to extract 2D slices from WORC datasets and save them as images (JPG/PDF).

Usage:
    python src/analysis/extract_worc_image.py --dataset lipo --sample-name Lipo-001 --output reports/thesis/figures/datasets/worc/lipo_example.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import SimpleITK as sitk
    import nibabel as nib
except ImportError:
    raise ImportError(
        "SimpleITK and nibabel packages are required. Install with: pip install SimpleITK nibabel"
    )

from src.classification_3d.utils.dataset_cleaning import find_valid_image_and_segmentation_files, natural_key


def extract_worc_3d_slice(dataset_name, sample_name=None, slice_idx=None, axis=0, output_path=None, format='jpg', dpi=300, grid_samples=None, data_path="datasets", preprocessing_method=None):
    """
    Extract 2D slices from WORC datasets and save them as images.
    Creates a visualization with a grid of smaller samples.
    
    Args:
        dataset_name (str): Name of the WORC dataset (e.g., 'lipo', 'gist', 'desmoid', 'liver')
        sample_name (str): Name of a specific sample (e.g., 'Lipo-001'). If None, uses first sample (default: None)
        slice_idx (int): Index of the slice to extract. If None, uses middle slice (default: None)
        axis (int): Axis along which to extract the slice (0=Z/axial, 1=Y/coronal, 2=X/sagittal). Default: 0 (axial)
        output_path (str): Path to save the image. If None, uses default name
        format (str): Output format ('jpg' or 'pdf'). Default: 'jpg'
        dpi (int): DPI for saved image. Default: 300
        grid_samples (int): Number of samples to show in grid. If None, shows 80 samples (default: None)
        data_path (str): Base path to datasets directory. Default: "datasets"
        preprocessing_method (str): Preprocessing method to use (e.g., 'median', 'mean', 'isotropic'). 
                                   If None, uses cleaned data directly. Default: None
    """
    # Map dataset names to cleaned dataset paths
    dataset_map = {
        'lipo': 'lipo_cleaned',
        'gist': 'gist_cleaned',
        'hcc': 'hcc_cleaned',
        'bflair': 'bflair_cleaned',
        'desmoid': 'desmoid_cleaned',
        'liver': 'liver_cleaned',
        'crlm': 'crlm_cleaned',
    }
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(
            f"Unknown WORC dataset: {dataset_name}. "
            f"Supported datasets: {list(dataset_map.keys())}"
        )
    
    cleaned_dataset_name = dataset_map[dataset_name.lower()]
    base_dataset_path = os.path.join(data_path, cleaned_dataset_name)
    
    # Determine dataset path - use preprocessed if specified
    if preprocessing_method:
        dataset_path = os.path.join(base_dataset_path, f"preprocessed_{preprocessing_method}")
        if not os.path.exists(dataset_path):
            raise ValueError(f"Preprocessed dataset path does not exist: {dataset_path}")
        print(f"Using preprocessed data: preprocessed_{preprocessing_method}")
    else:
        dataset_path = base_dataset_path
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Loading {dataset_name.upper()} dataset from {dataset_path}...")
    
    # Get all sample directories
    # Exclude preprocessed directories and other non-sample directories
    exclude_dirs = ["preprocessed", "cv_splits"]
    directory_names = [d for d in sorted(os.listdir(dataset_path), key=natural_key) 
                      if os.path.isdir(os.path.join(dataset_path, d)) and d not in exclude_dirs]
    
    if not directory_names:
        raise ValueError(f"No sample directories found in {dataset_path}")
    
    print(f"Found {len(directory_names)} samples")
    
    # Get image paths for all samples
    image_paths = []
    for data_point in directory_names:
        img_path, seg_path = find_valid_image_and_segmentation_files(dataset_path, data_point)
        if img_path and seg_path:
            image_paths.append((data_point, img_path))
        else:
            print(f"Warning: Could not find image/segmentation files in {data_point}")
    
    if not image_paths:
        raise ValueError(f"No valid image files found in {dataset_path}")
    
    print(f"Found {len(image_paths)} valid samples")
    
    # Load first sample to get dimensions
    first_sample_name, first_img_path = image_paths[0]
    print(f"\nLoading sample: {first_sample_name}")
    print(f"Image path: {first_img_path}")
    
    # Load NIfTI image
    img_sitk = sitk.ReadImage(first_img_path)
    img_array = sitk.GetArrayFromImage(img_sitk)  # Shape: (z, y, x)
    
    print(f"Image shape (z, y, x): {img_array.shape}")
    
    # Extract slice - for medical images, we typically want axial slices (z-axis)
    # SimpleITK GetArrayFromImage returns (z, y, x)
    # For visualization, we want to show axial slices (along z-axis) which gives us (y, x) slices
    if axis == 0:  # z-axis (axial slices - most common for medical imaging)
        if slice_idx is None:
            slice_idx = img_array.shape[0] // 2
        slice_2d = img_array[slice_idx, :, :]  # Shape: (y, x)
        axis_name = "Z (axial)"
    elif axis == 1:  # y-axis (coronal slices)
        if slice_idx is None:
            slice_idx = img_array.shape[1] // 2
        slice_2d = img_array[:, slice_idx, :]  # Shape: (z, x)
        axis_name = "Y (coronal)"
    elif axis == 2:  # x-axis (sagittal slices)
        if slice_idx is None:
            slice_idx = img_array.shape[2] // 2
        slice_2d = img_array[:, :, slice_idx]  # Shape: (z, y)
        axis_name = "X (sagittal)"
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")
    
    print(f"Extracted slice {slice_idx} along {axis_name} axis")
    print(f"Slice shape: {slice_2d.shape}")
    
    # Grid of smaller samples
    if grid_samples is None:
        grid_samples = min(25, len(image_paths))  # Show up to 25 samples (5 rows x 5 cols)
    
    # Calculate grid dimensions
    n_cols = 5
    n_rows = (grid_samples + n_cols - 1) // n_cols  # Ceiling division
    
    # Limit to reasonable size
    n_rows = min(n_rows, 5)
    grid_samples = min(grid_samples, n_rows * n_cols)
    
    print(f"Creating grid with {grid_samples} samples ({n_rows} rows x {n_cols} cols)")
    
    # Create figure with grid layout
    # Adjust figure size based on number of rows
    fig_width = 12
    fig_height = max(8, 1.2 * n_rows)  # Minimum height of 8, scale with rows
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Create grid layout
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.1, wspace=-0.7)
    
    # Sample indices for grid (diverse selection)
    grid_indices = np.linspace(0, len(image_paths) - 1, grid_samples, dtype=int)
    
    for idx, grid_idx in enumerate(grid_indices):
        row = idx // n_cols
        col = idx % n_cols
        
        if row >= n_rows:  # Don't exceed figure bounds
            break
        
        sample_name_grid, img_path_grid = image_paths[grid_idx]
        
        try:
            # Load NIfTI image
            img_sitk_grid = sitk.ReadImage(img_path_grid)
            img_array_grid = sitk.GetArrayFromImage(img_sitk_grid)  # Shape: (z, y, x)
            
            # Extract middle slice along same axis
            if axis == 0:  # z-axis (axial)
                slice_grid = img_array_grid[img_array_grid.shape[0] // 2, :, :]  # Shape: (y, x)
            elif axis == 1:  # y-axis (coronal)
                slice_grid = img_array_grid[:, img_array_grid.shape[1] // 2, :]  # Shape: (z, x)
            else:  # axis == 2 (sagittal)
                slice_grid = img_array_grid[:, :, img_array_grid.shape[2] // 2]  # Shape: (z, y)
            
            # Normalize to [0, 1] using percentile-based normalization to handle outliers
            # This is more robust than min-max normalization
            p2 = np.percentile(slice_grid, 2)
            p98 = np.percentile(slice_grid, 98)
            if p98 > p2:
                slice_grid_norm = np.clip((slice_grid - p2) / (p98 - p2), 0, 1)
            else:
                # Fallback to min-max if percentiles are too close
                slice_min = slice_grid.min()
                slice_max = slice_grid.max()
                if slice_max > slice_min:
                    slice_grid_norm = (slice_grid - slice_min) / (slice_max - slice_min)
                else:
                    slice_grid_norm = slice_grid
            
            # Create subplot for this grid cell
            ax_grid = fig.add_subplot(gs[row, col])
            ax_grid.imshow(slice_grid_norm, cmap='gray', interpolation='bilinear', aspect='equal')
            ax_grid.axis('off')
        except Exception as e:
            print(f"Warning: Could not load {sample_name_grid}: {e}")
            # Create empty subplot
            ax_grid = fig.add_subplot(gs[row, col])
            ax_grid.axis('off')
    
    # No title in image - caption will be added in LaTeX
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        output_dir = Path("reports/thesis/figures/datasets/worc")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_name}_example.{format}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image
    if format.lower() == 'pdf':
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=dpi)
    elif format.lower() in ['jpg', 'jpeg']:
        plt.savefig(output_path, format='jpg', bbox_inches='tight', dpi=dpi, quality=95)
    elif format.lower() == 'png':
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=dpi)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'jpg', 'png', or 'pdf'.")
    
    print(f"\nImage saved to: {output_path}")
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2D slices from WORC datasets and save them as images."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['lipo', 'gist', 'desmoid', 'liver', 'crlm', 'hcc', 'bflair'],
        help='Name of the WORC dataset'
    )
    parser.add_argument(
        '--sample-name',
        type=str,
        default=None,
        help='Name of a specific sample (e.g., Lipo-001). If None, uses first sample (default: None)'
    )
    parser.add_argument(
        '--slice-idx',
        type=int,
        default=None,
        help='Index of the slice to extract. If None, uses middle slice (default: None)'
    )
    parser.add_argument(
        '--axis',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Axis along which to extract the slice (0=Z/axial, 1=Y/coronal, 2=X/sagittal). Default: 0 (axial slices)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the image. If None, uses default name in reports/thesis/figures/datasets/worc/'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='jpg',
        choices=['jpg', 'jpeg', 'png', 'pdf'],
        help='Output format (default: jpg)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved image (default: 300)'
    )
    parser.add_argument(
        '--grid-samples',
        type=int,
        default=None,
        help='Number of samples to show in grid. If None, shows 25 samples (5 rows x 5 cols) (default: None)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='datasets',
        help='Base path to datasets directory (default: datasets)'
    )
    parser.add_argument(
        '--preprocessing-method',
        type=str,
        default=None,
        help='Preprocessing method to use (e.g., median, mean, isotropic, volumetric_isotropic). If not specified, uses cleaned data directly (default: None)'
    )
    
    args = parser.parse_args()
    
    # Validate preprocessing method if provided
    if args.preprocessing_method is not None:
        valid_methods = ['median', 'mean', 'isotropic', 'volumetric_isotropic']
        if args.preprocessing_method not in valid_methods:
            raise ValueError(
                f"Invalid preprocessing method: {args.preprocessing_method}. "
                f"Valid methods: {valid_methods}"
            )
    
    extract_worc_3d_slice(
        dataset_name=args.dataset,
        sample_name=args.sample_name,
        slice_idx=args.slice_idx,
        axis=args.axis,
        output_path=args.output,
        format=args.format,
        dpi=args.dpi,
        grid_samples=args.grid_samples,
        data_path=args.data_path,
        preprocessing_method=args.preprocessing_method
    )


if __name__ == "__main__":
    main()

"""
# preprocessed median
python -m src.analysis.extract_worc_image --dataset desmoid --preprocessing-method median --format pdf --output reports/thesis/figures/datasets/worc/desmoid_example.pdf
python -m src.analysis.extract_worc_image --dataset lipo --preprocessing-method median --format pdf --output reports/thesis/figures/datasets/worc/lipo_example.pdf
python -m src.analysis.extract_worc_image --dataset gist --preprocessing-method median --format pdf --output reports/thesis/figures/datasets/worc/gist_example.pdf
python -m src.analysis.extract_worc_image --dataset liver --preprocessing-method median --format pdf --output reports/thesis/figures/datasets/worc/liver_example.pdf
python -m src.analysis.extract_worc_image --dataset crlm --preprocessing-method median --format pdf --output reports/thesis/figures/datasets/worc/crlm_example.pdf

# unprocessed
python -m src.analysis.extract_worc_image --dataset desmoid --format pdf --output reports/thesis/figures/datasets/worc/desmoid_example_unprocessed.pdf
python -m src.analysis.extract_worc_image --dataset lipo --format pdf --output reports/thesis/figures/datasets/worc/lipo_example_unprocessed.pdf
python -m src.analysis.extract_worc_image --dataset gist --format pdf --output reports/thesis/figures/datasets/worc/gist_example_unprocessed.pdf
python -m src.analysis.extract_worc_image --dataset liver --format pdf --output reports/thesis/figures/datasets/worc/liver_example_unprocessed.pdf
python -m src.analysis.extract_worc_image --dataset crlm --format pdf --output reports/thesis/figures/datasets/worc/crlm_example_unprocessed.pdf
"""
