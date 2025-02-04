"""
Data loading and preprocessing module for medical image datasets.
This module provides functionality for loading and preparing medical image data
for deep learning models, including custom Dataset classes and data loaders.
"""

import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WORCDataset(Dataset):
    """
    Custom Dataset class for WORC medical image data.

    Args:
        data (list): List of preprocessed image tensors
        labels (list): List of corresponding labels
        transform (callable, optional): Optional transform to be applied on images
    """

    def __init__(self, data, labels, transform=None):
        """Initialize dataset with images and labels."""
        self.data = data
        self.labels = labels
        self.transform = transform

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
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


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
    images = []
    labels = []

    # Get all patient folders and sort them for reproducibility
    patient_folders = sorted(
        glob.glob(os.path.join(dataset_path, f"{name.capitalize()}-*"))
    )

    if not patient_folders:
        raise ValueError("No patient folders found in " + dataset_path)

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

    # Check class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")

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


def get_data_loaders(
    dataset_name, num_workers, batch_size, split="train", data_path="datasets"
):
    """
    Create data loaders for the specified dataset split.

    Args:
        dataset_name (str): Name of the dataset ('lipo', 'desmoid', 'gist')
        num_workers (int): Number of worker processes for data loading
        batch_size (int): Batch size for the data loaders
        split (str, optional): Which split to load ('train' or 'test'). Defaults to 'train'
        data_path (str): Base path to the datasets directory. Defaults to 'datasets'

    Returns:
        tuple: Data loaders and number of classes
    """
    # ImageNet normalization
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    try:
        dataset = load_dataset(dataset_name, data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {str(e)}") from e

    if split == "train":
        # Create train and validation datasets
        train_dataset = WORCDataset(
            dataset["train_data"], dataset["train_labels"], transform=transform
        )
        val_dataset = WORCDataset(
            dataset["val_data"], dataset["val_labels"], transform=transform
        )

        # Add prefetch factor for better data loading performance
        loader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,  # Keep workers alive between epochs
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

        return train_loader, val_loader, dataset["num_classes"]

    if split == "test":
        # Create test dataset
        test_dataset = WORCDataset(
            dataset["test_data"], dataset["test_labels"], transform=transform
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
