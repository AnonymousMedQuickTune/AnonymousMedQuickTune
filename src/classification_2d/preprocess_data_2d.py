import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils.common_utils import yaml_to_neps_pipeline_space

import glob
import os
import pickle
import random

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import KFold

# TODO: Use other dataset(s)

class WORCDataset(Dataset):
    """
    Custom Dataset class for WORC medical image data.

    Args:
        data (list): List of preprocessed image tensors
        labels (list): List of corresponding labels
        transform (callable, optional): Optional transform to be applied on images
        is_training (bool): Whether this is a training dataset
        augmentation_type (str): Type of augmentation to use ('medical' or 'trivial')
        normalization_stats (dict, optional): Dictionary containing 'mean' and 'std' for normalization
    """

    def __init__(self, data, labels, transform=None, is_training=False, augmentation_type='medical', normalization_stats=None):
        """Initialize dataset with images and labels."""
        self.data = data
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        self.normalization_stats = normalization_stats
        
        # Set up augmentation based on type
        if is_training:
            if augmentation_type == 'medical':
                self.augmentation = MedicalImageAugmentation(p=0.5)
                print("\n\n\nUsing medical augmentation\n\n\n")
            elif augmentation_type == 'trivial':
                self.augmentation = transforms.TrivialAugmentWide()
                print("\n\n\nUsing trivial augmentation\n\n\n")
            else:
                raise ValueError(f"Unknown augmentation type: {augmentation_type}")
        else:
            self.augmentation = None

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image, label) where image is a tensor and label is an int
        """
        image = self.data[idx]

        # Apply augmentation during training
        if self.is_training and self.augmentation is not None:
            if isinstance(self.augmentation, MedicalImageAugmentation):
                # Medical augmentation expects float32 (0-1 range)
                image = self.augmentation(image)
            else:  # TrivialAugmentWide
                # Convert to uint8 (0-255 range) for TrivialAugmentWide
                image = (image * 255).byte()
                image = self.augmentation(image)
                # Convert back to float32 (0-1 range)
                image = image.float() / 255.0

        # Apply normalization if stats are provided
        if self.normalization_stats is not None:
            mean = self.normalization_stats['mean']
            std = self.normalization_stats['std']
            if isinstance(mean, (int, float)):
                mean = [mean] * image.shape[0]  # Broadcast to all channels
            if isinstance(std, (int, float)):
                std = [std] * image.shape[0]  # Broadcast to all channels
            transform = transforms.Normalize(mean=mean, std=std)
            image = transform(image)

        # Apply other transforms
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class MedicalImageAugmentation:
    """
    Advanced augmentation pipeline for medical images, specifically designed for classification tasks.
    Implements state-of-the-art augmentation techniques that preserve medical image characteristics.
    """

    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of applying each augmentation
        """
        self.p = p

    def __call__(self, img):
        """
        Apply augmentations to the image with probability p.

        Args:
            img (torch.Tensor): Input image tensor of shape [C, H, W]

        Returns:
            torch.Tensor: Augmented image
        """
        # Convert to PIL for some transformations
        img_np = img.permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_np * 255).astype("uint8"))

        # Random rotation (small angles to preserve medical relevance)
        if random.random() < self.p:
            angle = random.uniform(-10, 10)
            img_pil = transforms.functional.rotate(img_pil, angle, fill=0)

        # Random affine transformation (slight deformation)
        if random.random() < self.p:
            scale = random.uniform(0.95, 1.05)
            translate = (random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02))
            img_pil = transforms.functional.affine(
                img_pil, angle=0, translate=translate, scale=scale, shear=0, fill=0
            )

        # Convert back to tensor
        img = transforms.functional.to_tensor(img_pil)

        # Gaussian noise
        if random.random() < self.p:
            noise = torch.randn_like(img) * 0.02
            img = torch.clamp(img + noise, 0, 1)

        # Random gamma correction
        if random.random() < self.p:
            gamma = random.uniform(0.8, 1.2)
            img = torch.pow(img, gamma)

        # Random contrast
        if random.random() < self.p:
            contrast_factor = random.uniform(0.9, 1.1)
            img = transforms.functional.adjust_contrast(img, contrast_factor)

        # Random brightness
        if random.random() < self.p:
            brightness_factor = random.uniform(0.9, 1.1)
            img = transforms.functional.adjust_brightness(img, brightness_factor)

        # Elastic deformation
        if random.random() < self.p:
            img = self._elastic_transform(img)

        return img

    def _elastic_transform(self, img, alpha=1000, sigma=30):
        """
        Apply elastic deformation to image.

        Args:
            img (torch.Tensor): Input image
            alpha (float): Scaling factor for displacement
            sigma (float): Gaussian filter parameter
        """
        shape = img.shape[1:]
        dx = torch.randn(shape) * alpha
        dy = torch.randn(shape) * alpha

        # Apply Gaussian blur to the displacement fields
        dx = transforms.GaussianBlur(kernel_size=7, sigma=sigma)(
            dx.unsqueeze(0)
        ).squeeze()
        dy = transforms.GaussianBlur(kernel_size=7, sigma=sigma)(
            dy.unsqueeze(0)
        ).squeeze()

        x, y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
        indices = torch.stack([y + dy, x + dx])

        # Normalize indices to [-1, 1] for grid_sample
        indices[0] = 2 * indices[0] / (shape[0] - 1) - 1
        indices[1] = 2 * indices[1] / (shape[1] - 1) - 1

        # Reshape indices for grid_sample
        grid = indices.permute(1, 2, 0).unsqueeze(0)

        # Apply transformation
        img = F.grid_sample(
            img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros"
        ).squeeze(0)
        return img


def load_2d_dataset(name, data_path="datasets", seed=42):
    """
    Load and preprocess a medical image dataset.

    Args:
        name (str): Name of the dataset to load ('lipo', 'desmoid', 'gist')
        data_path (str): Base path to the datasets directory. Defaults to 'datasets'

    Returns:
        dict: Dictionary containing dataset splits and metadata
    """
    name = name.lower()
    if name not in ["lipo", "desmoid", "gist"]:
        raise ValueError(f"Unknown dataset: {name}")

    dataset_path = os.path.join(data_path, name)

    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset directory not found: {dataset_path}")

    images = []
    labels = []

    # Try different folder patterns
    patterns = [
        f"{name.capitalize()}-*",  # e.g., "Lipo-001"
        f"{name.upper()}-*",  # e.g., "GIST-001"
        f"{name}-*",  # e.g., "gist-001"
        "*",  # any folder in the dataset directory
    ]

    patient_folders = []
    for pattern in patterns:
        folders = sorted(glob.glob(os.path.join(dataset_path, pattern)))
        if folders:
            patient_folders = folders
            break

    if not patient_folders:
        raise ValueError(
            f"No patient folders found in {dataset_path} using patterns: {patterns}"
        )

    print(f"Found {len(patient_folders)} patients in {name} dataset")

    # Load all images and labels
    for folder in patient_folders:
        try:
            # Load label
            label_file = glob.glob(os.path.join(folder, "*_label.csv"))[0]
            label = pd.read_csv(label_file)["Diagnosis_binary"].iloc[0]

            # Load and process image
            img_path = os.path.join(folder, "image.nii.gz")
            nifti_img = nib.load(img_path)
            img_data = nifti_img.get_fdata()

            # Get middle slice
            middle_slice_idx = img_data.shape[2] // 2
            slice_data = img_data[:, :, middle_slice_idx]

            # Normalize slice
            slice_min, slice_max = slice_data.min(), slice_data.max()
            if slice_max > slice_min:
                slice_data = (slice_data - slice_min) / (slice_max - slice_min)

            # Convert to RGB and tensor
            rgb_slice = np.stack([slice_data] * 3, axis=0)
            tensor_slice = torch.FloatTensor(rgb_slice)

            # Resize if needed
            if tensor_slice.shape[1:] != (224, 224):
                tensor_slice = transforms.Resize((224, 224))(tensor_slice)

            images.append(tensor_slice)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")
            continue

    if not images:
        raise ValueError("No valid images were loaded")

    # Convert labels to numpy array
    labels = np.array(labels)

    # Remove classes with too few samples (less than 2)
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_labels = unique_labels[counts >= 2]

    # Filter out samples with invalid labels
    valid_mask = np.isin(labels, valid_labels)
    images = [img for i, img in enumerate(images) if valid_mask[i]]
    labels = labels[valid_mask]

    # Recheck class distribution after filtering
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution after filtering: {dict(zip(unique_labels, counts))}")

    # Split into train+val and test (80-20)
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    print(
        f"\nDataset split (train+val/test): {len(train_val_data)}/{len(test_data)}"
    )

    return {
        "train_val_data": train_val_data,
        "train_val_labels": train_val_labels,
        "test_data": test_data,
        "test_labels": test_labels,
        "num_classes": len(unique_labels),
    }


def calculate_normalization_stats(images):
    """
    Calculate mean and std across all images in the dataset.
    This function should only be called with training images to prevent data leakage.

    Args:
        images (list): List of image tensors of shape [C, H, W] from training set only

    Returns:
        tuple: (means, stds) for each channel
    """
    # Stack all images into a single tensor [N, C, H, W]
    all_images = torch.stack(images)

    # Calculate mean and std across all images for each channel
    means = torch.mean(all_images, dim=[0, 2, 3])
    stds = torch.std(all_images, dim=[0, 2, 3])

    return means.tolist(), stds.tolist()


def get_kfold_loaders(
    data, 
    labels, 
    k_folds, 
    batch_size, 
    num_workers, 
    fold_idx,
    normalization_stats=None,
    data_path="datasets",
    augmentation_type='medical'
):
    """
    Create data loaders for k-fold cross validation.

    Args:
        data (list): Combined training and validation data
        labels (numpy.ndarray): Combined training and validation labels
        k_folds (int): Number of folds for cross-validation
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        fold_idx (int): Current fold index
        normalization_stats (dict, optional): Pre-computed normalization statistics
        data_path (str): Path to dataset directory
        augmentation_type (str): Type of augmentation to use

    Returns:
        tuple: (train_loader, val_loader) for the current fold
    """
    

    # Create k-fold splitter
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Convert data list to indices
    indices = np.arange(len(data))

    # Get train and validation indices for current fold
    for i, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if i == fold_idx:
            break

    # Split data for current fold
    train_data = [data[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_data = [data[i] for i in val_idx]
    val_labels = labels[val_idx]

    # Calculate normalization stats from training data if not provided
    if normalization_stats is None:
        mean = np.mean([img.numpy().mean() for img in train_data])
        std = np.std([img.numpy().std() for img in train_data])
        normalization_stats = {'mean': mean, 'std': std}


    # Create datasets
    train_dataset = WORCDataset(
        train_data,
        train_labels,
        normalization_stats=normalization_stats,
        augmentation_type=augmentation_type,
        is_training=True
    )
    
    val_dataset = WORCDataset(
        val_data,
        val_labels,
        normalization_stats=normalization_stats,
        augmentation_type=None,  # No augmentation for validation
        is_training=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def get_max_batch_size(pipeline_space):
    batch_size = pipeline_space.get('batch_size', None)
    if batch_size is None:
        return 32
    return batch_size.upper

@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="main_experiment_config.yaml",
)
def preprocess_and_cache_datasets(config: DictConfig) -> None:
    """
    Preprocess and cache datasets for faster experiment initialization.

    Args:
        config (DictConfig): Hydra configuration object
    """
    print("\nPreprocessing datasets...")

    # Get dataset name from config
    dataset = config.data.dataset
    print(f"Processing dataset: {dataset}")

    # Create cache directory in the same location as the dataset
    data_path = Path(config.data.path)
    cache_dir = data_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache filename - simplified to just use dataset name
    # Convert YAML pipeline space configuration into NePS-compatible format
    # NePS requires a specific dictionary structure for hyperparameter definitions
    pipeline_space = yaml_to_neps_pipeline_space(config.pipeline_space)
    cache_file = cache_dir / f"{config.data.dataset}_bs{get_max_batch_size(pipeline_space)}.pkl"

    if cache_file.exists():
        print(f"Cache file already exists at {cache_file}")
        print("Delete it manually if you want to regenerate the cache.")
        return

    # Load raw dataset first
    print(f"Loading dataset '{dataset}'...")
    dataset_dict = load_2d_dataset(dataset, data_path=config.data.path, seed=config.seed)

    # Calculate normalization statistics from training data only
    print("Calculating dataset-specific normalization statistics...")
    means, stds = calculate_normalization_stats(dataset_dict["train_val_data"])
    print(f"Dataset means: {means}")
    print(f"Dataset stds: {stds}")

    # Add normalization stats to dataset_dict
    dataset_dict["normalization_stats"] = (means, stds)

    # Verify all required keys are present
    required_keys = [
        "train_val_data", 
        "train_val_labels", 
        "test_data", 
        "test_labels", 
        "num_classes",
        "normalization_stats"
    ]
    missing_keys = [key for key in required_keys if key not in dataset_dict]
    if missing_keys:
        raise KeyError(f"Dataset dictionary missing required keys: {missing_keys}")

    # Cache the complete dataset dictionary
    print(f"\nSaving cache to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(dataset_dict, f)

    print("\nPreprocessing completed!")
    print(f"Dataset '{dataset}' preprocessed and cached with {dataset_dict['num_classes']} classes")
    print(f"Dataset-specific normalization values have been calculated and cached.")

if __name__ == "__main__":
    preprocess_and_cache_datasets()
