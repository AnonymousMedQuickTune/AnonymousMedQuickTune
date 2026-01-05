#!/usr/bin/env python3
"""
Script to extract a 2D slice from a MedMNIST 3D dataset and save it as an image (JPG/PDF).

Usage:
    python scripts/extract_medmnist_image.py --dataset organmnist3d --output reports/thesis/figures/organmnist3d_example.jpg
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    raise ImportError(
        "medmnist package is required. Install it with: pip install medmnist"
    )


def extract_medmnist_3d_slice(dataset_name, sample_idx=0, slice_idx=None, axis=2, output_path=None, format='jpg', dpi=300, grid_samples=None):
    """
    Extract a 2D slice from a MedMNIST 3D dataset and save it as an image.
    Creates a visualization with a large example image and a grid of smaller samples.
    
    Args:
        dataset_name (str): Name of the MedMNIST 3D dataset (e.g., 'organmnist3d')
        sample_idx (int): Index of the main sample to extract (default: 0)
        slice_idx (int): Index of the slice to extract. If None, uses middle slice (default: None)
        axis (int): Axis along which to extract the slice (0=H, 1=W, 2=D). Default: 2 (depth)
        output_path (str): Path to save the image. If None, uses default name
        format (str): Output format ('jpg' or 'pdf'). Default: 'jpg'
        dpi (int): DPI for saved image. Default: 300
        grid_samples (int): Number of samples to show in grid. If None, shows 80 samples (default: None)
    """
    # Map dataset names
    dataset_map = {
        'organmnist3d': ('OrganMNIST3D', 'organmnist3d'),
        'nodulemnist3d': ('NoduleMNIST3D', 'nodulemnist3d'),
        'adrenalmnist3d': ('AdrenalMNIST3D', 'adrenalmnist3d'),
        'fracturemnist3d': ('FractureMNIST3D', 'fracturemnist3d'),
        'vesselmnist3d': ('VesselMNIST3D', 'vesselmnist3d'),
        'synapsemnist3d': ('SynapseMNIST3D', 'synapsemnist3d'),
    }
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(
            f"Unknown MedMNIST 3D dataset: {dataset_name}. "
            f"Supported datasets: {list(dataset_map.keys())}"
        )
    
    dataset_class_name, info_key = dataset_map[dataset_name.lower()]
    
    # Get dataset info
    info = INFO[info_key]
    class_names = info['label']
    
    print(f"Loading {dataset_class_name} dataset...")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Import and load dataset
    dataset_module = getattr(medmnist, dataset_class_name)
    dataset = dataset_module(split='train', download=False, root="datasets", as_rgb=False)
    
    if sample_idx >= len(dataset):
        raise ValueError(f"Sample index {sample_idx} >= dataset size {len(dataset)}")
    
    # Load sample
    img, label = dataset[sample_idx]
    
    # Convert label to integer
    if isinstance(label, np.ndarray):
        label = int(label.item() if label.size == 1 else label[0])
    else:
        label = int(label)
    
    # Convert to numpy array
    if hasattr(img, 'numpy'):
        img = img.numpy()
    elif hasattr(img, 'array'):
        img = np.array(img)
    else:
        img = np.array(img)
    
    # Ensure shape is (H, W, D) - remove channel dimension if present
    if len(img.shape) == 4:  # (C, H, W, D)
        img = img[0]  # Take first channel
    elif len(img.shape) != 3:  # Should be (H, W, D)
        raise ValueError(f"Unexpected image shape: {img.shape}")
    
    print(f"\nImage shape: {img.shape}")
    print(f"Label: {label} ({class_names[str(label)] if str(label) in class_names else 'unknown'})")
    
    # Extract slice
    if slice_idx is None:
        # Use middle slice
        slice_idx = img.shape[axis] // 2
    
    if axis == 0:
        slice_2d = img[slice_idx, :, :]
        axis_name = "H (height)"
    elif axis == 1:
        slice_2d = img[:, slice_idx, :]
        axis_name = "W (width)"
    elif axis == 2:
        slice_2d = img[:, :, slice_idx]
        axis_name = "D (depth)"
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")
    
    print(f"Extracted slice {slice_idx} along {axis_name} axis")
    print(f"Slice shape: {slice_2d.shape}")
    
    # Grid of smaller samples only
    if grid_samples is None:
        grid_samples = min(56, len(dataset))  # Show up to 56 samples (7 rows x 8 cols)
    
    # Calculate grid dimensions
    n_cols = 8
    n_rows = (grid_samples + n_cols - 1) // n_cols  # Ceiling division
    
    # Limit to reasonable size
    n_rows = min(n_rows, 10)
    grid_samples = min(grid_samples, n_rows * n_cols)
    
    print(f"Creating grid with {grid_samples} samples ({n_rows} rows x {n_cols} cols)")
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(12, 1.5 * n_rows), dpi=dpi)
    
    # Create grid layout
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.05, wspace=0.05)
    
    # Sample indices for grid (diverse selection)
    grid_indices = np.linspace(0, len(dataset) - 1, grid_samples, dtype=int)
    
    for idx, grid_idx in enumerate(grid_indices):
        row = idx // n_cols
        col = idx % n_cols
        
        if row >= n_rows:  # Don't exceed figure bounds
            break
            
        # Load sample for grid
        img_grid, label_grid = dataset[grid_idx]
        
        # Convert to numpy
        if hasattr(img_grid, 'numpy'):
            img_grid = img_grid.numpy()
        elif hasattr(img_grid, 'array'):
            img_grid = np.array(img_grid)
        else:
            img_grid = np.array(img_grid)
        
        # Ensure shape is (H, W, D)
        if len(img_grid.shape) == 4:
            img_grid = img_grid[0]
        
        # Extract middle slice along same axis
        if axis == 0:
            slice_grid = img_grid[img_grid.shape[0] // 2, :, :]
        elif axis == 1:
            slice_grid = img_grid[:, img_grid.shape[1] // 2, :]
        else:  # axis == 2
            slice_grid = img_grid[:, :, img_grid.shape[2] // 2]
        
        # Normalize
        slice_grid_norm = (slice_grid - slice_grid.min()) / (slice_grid.max() - slice_grid.min() + 1e-8)
        
        # Create subplot for this grid cell
        ax_grid = fig.add_subplot(gs[row, col])
        ax_grid.imshow(slice_grid_norm, cmap='gray', interpolation='nearest')
        ax_grid.axis('off')
    
    # No title in image - caption will be added in LaTeX
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        output_dir = Path("reports/thesis/figures/datasets/medmnist3d")
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
        description="Extract a 2D slice from a MedMNIST 3D dataset and save it as an image."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='organmnist3d',
        choices=['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d'],
        help='Name of the MedMNIST 3D dataset'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Index of the sample to extract (default: 0)'
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
        default=2,
        choices=[0, 1, 2],
        help='Axis along which to extract the slice (0=H, 1=W, 2=D). Default: 2 (depth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the image. If None, uses default name in reports/thesis/figures/'
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
        help='Number of samples to show in grid. If None, shows 80 samples (default: None)'
    )
    
    args = parser.parse_args()
    
    extract_medmnist_3d_slice(
        dataset_name=args.dataset,
        sample_idx=args.sample_idx,
        slice_idx=args.slice_idx,
        axis=args.axis,
        output_path=args.output,
        format=args.format,
        dpi=args.dpi,
        grid_samples=args.grid_samples
    )


if __name__ == "__main__":
    main()
    # example: python src/analysis/extract_medmnist_image.py --dataset synapsemnist3d --format pdf --output reports/thesis/figures/datasets/medmnist3d/synapsemnist3d_example.pdf 

