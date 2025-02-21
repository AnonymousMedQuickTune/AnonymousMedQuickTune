"""
Data loading and preprocessing module for medical image datasets.
This module provides functionality for loading and preparing medical image data
for deep learning models, including custom Dataset classes and data loaders.
"""

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
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WORCDataset(Dataset):
    """
    Custom Dataset class for WORC medical image data.

    Args:
        data (list): List of preprocessed image tensors
        labels (list): List of corresponding labels
        transform (callable, optional): Optional transform to be applied on images
        is_training (bool): Whether this is a training dataset
    """

    def __init__(self, data, labels, transform=None, is_training=False):
        """Initialize dataset with images and labels."""
        self.data = data
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        self.augmentation = MedicalImageAugmentation(p=0.5) if is_training else None

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
            image = self.augmentation(image)

        # Apply normalization or other transforms
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


def load_dataset(name, data_path="datasets"):
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

    # First split into train and temp (80-20)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Split temp into val and test (50-50, resulting in 80-10-10 split)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(
        f"\nDataset split (train/val/test): {len(train_data)}/{len(val_data)}/{len(test_data)}"
    )

    return {
        "train_data": train_data,
        "train_labels": train_labels,
        "val_data": val_data,
        "val_labels": val_labels,
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


def get_data_loaders(
    dataset_name,
    num_workers,
    batch_size,
    split="train",
    data_path="datasets",
    normalization_stats=None,
):
    """
    Create data loaders for the specified dataset split.

    Args:
        dataset_name (str): Name of the dataset ('lipo', 'desmoid', 'gist')
        num_workers (int): Number of worker processes for data loading
        batch_size (int): Batch size for the data loaders
        split (str, optional): Which split to load ('train' or 'test'). Defaults to 'train'
        data_path (str, optional): Base path to the datasets directory. Default to 'datasets'
        normalization_stats (tuple, optional): Normalization stats for the dataset. If not provided,
            stats will be calculated from training data if split is 'train'.
            For 'test' split, stats must be provided.

    Returns:
        tuple: (train_loader, val_loader, num_classes) if split is 'train'
        tuple: (test_loader, num_classes) if split is 'test'
    """
    try:
        # Load dataset first
        dataset = load_dataset(dataset_name, data_path)

        # Try to load normalization stats from cache if not provided
        if normalization_stats is None:
            cache_dir = os.path.join(data_path, "cache")
            cache_file = os.path.join(
                cache_dir, f"{dataset_name}_normalization_stats.pkl"
            )

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    normalization_stats = cached_data["normalization_stats"]
            elif split == "train":
                # Calculate stats only from training data to prevent data leakage
                means, stds = calculate_normalization_stats(dataset["train_data"])
                normalization_stats = (means, stds)

                # Cache the normalization stats
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump({"normalization_stats": normalization_stats}, f)
            else:
                raise ValueError(
                    "normalization_stats must be provided for test split or available in cache "
                    "to ensure consistent normalization with training data"
                )

        means, stds = normalization_stats

        # For debugging / testing:
        # ImageNet normalization stats
        # means = [0.485, 0.456, 0.406]
        # stds = [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=means, std=stds)

        if split == "train":
            # Create train and validation datasets
            train_dataset = WORCDataset(
                dataset["train_data"],
                dataset["train_labels"],
                transform=normalize,
                is_training=True,  # Enable augmentation for training
            )
            val_dataset = WORCDataset(
                dataset["val_data"],
                dataset["val_labels"],
                transform=normalize,
                is_training=False,  # Disable augmentation for validation
            )

            # Add prefetch factor for better data loading performance
            loader_args = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": True,
                "prefetch_factor": 2,
                "persistent_workers": True,
            }

            train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

            return train_loader, val_loader, dataset["num_classes"]

        if split == "test":
            test_dataset = WORCDataset(
                dataset["test_data"],
                dataset["test_labels"],
                transform=normalize,
                is_training=False,  # Disable augmentation for test
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            return test_loader, dataset["num_classes"]

        raise ValueError(f"Unknown split: {split}")

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {str(e)}") from e


def get_kfold_loaders(
    data, 
    labels, 
    k_folds, 
    batch_size, 
    num_workers, 
    fold_idx,
    normalization_stats=None,
    data_path="datasets"
):
    """
    Create train and validation loaders for a specific fold.
    Normalization statistics are calculated only from training data of the current fold.
    """
    # Create k-fold splitter
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Convert to list for indexing
    data_list = list(data)
    labels_list = list(labels)

    # Get the splits for the current fold
    splits = list(kfold.split(data_list))
    train_idx, val_idx = splits[fold_idx]

    # Get training data for this fold
    train_data = [data_list[i] for i in train_idx]
    train_labels = [labels_list[i] for i in train_idx]
    
    # Calculate normalization stats from training data only if not provided
    if normalization_stats is None:
        means, stds = calculate_normalization_stats(train_data)
        normalization_stats = (means, stds)
    
    means, stds = normalization_stats
    normalize = transforms.Normalize(mean=means, std=stds)

    # Create datasets for this fold
    train_dataset = WORCDataset(
        train_data,
        train_labels,
        transform=normalize,
        is_training=True,
    )

    val_dataset = WORCDataset(
        [data_list[i] for i in val_idx],
        [labels_list[i] for i in val_idx],
        transform=normalize,
        is_training=False,
    )

    # Add prefetch factor for better data loading performance
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args
    )

    return train_loader, val_loader
