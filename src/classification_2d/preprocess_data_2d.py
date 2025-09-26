import os
import pickle
import random
import shutil
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.common_utils import yaml_to_neps_pipeline_space


class BrainTumorDataset(Dataset):
    """
    Custom Dataset class for Brain Tumor MRI data.

    Args:
        data (list): List of preprocessed image tensors
        labels (list): List of corresponding labels
        is_training (bool): Whether this is a training dataset
        augmentation_type (str): Type of augmentation to use ('medical' or 'trivial')
        normalization_stats (dict, optional): Dictionary containing 'mean' and 'std' for normalization
    """

    def __init__(
        self,
        data,
        labels,
        is_training=False,
        augmentation_type="medical",
        normalization_stats=None,
    ):
        self.data = data
        self.labels = labels
        self.is_training = is_training
        self.normalization_stats = normalization_stats

        # Set up augmentation based on type
        if is_training:
            if augmentation_type == "medical":
                self.augmentation = MedicalBrainAugmentation(p=0.5)
                print("\nUsing medical brain tumor augmentation")
            elif augmentation_type == "trivial":
                self.augmentation = transforms.TrivialAugmentWide()
                print("\nUsing trivial augmentation")
            else:
                raise ValueError(f"Unknown augmentation type: {augmentation_type}")
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        # Apply augmentation during training
        if self.is_training and self.augmentation is not None:
            # Check which type of augmentation we're using
            if isinstance(self.augmentation, MedicalBrainAugmentation):
                # MedicalBrainAugmentation expects float tensors in range [0,1]
                # and handles the transformations internally
                image = self.augmentation(image)
            else:  # TrivialAugmentWide
                # TrivialAugmentWide expects byte tensors in range [0,255]
                image = (image * 255).byte()  # Convert from [0,1] float to [0,255] byte
                image = self.augmentation(image)
                image = image.float() / 255.0  # Convert back to [0,1] float range

        # Apply normalization if stats are provided
        if self.normalization_stats is not None:
            mean = self.normalization_stats["mean"]
            std = self.normalization_stats["std"]
            normalize_transform = transforms.Normalize(mean=mean, std=std)
            image = normalize_transform(image)

        return image, self.labels[idx]


class MedicalBrainAugmentation:
    """
    Specialized augmentation pipeline for brain MRI images.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Apply brain MRI-specific augmentations.
        """
        # Convert to PIL for some transformations
        img_np = img.permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_np * 255).astype("uint8"))

        # Gentle rotation (brain scans shouldn't be rotated too much)
        if random.random() < self.p:
            angle = random.uniform(-5, 5)  # Reduced rotation range
            img_pil = transforms.functional.rotate(img_pil, angle, fill=0)

        # Slight zoom/scale variation
        if random.random() < self.p:
            scale = random.uniform(0.98, 1.02)  # Reduced scale range
            img_pil = transforms.functional.affine(
                img_pil, angle=0, translate=(0, 0), scale=scale, shear=0, fill=0
            )

        # Convert back to tensor
        img = transforms.functional.to_tensor(img_pil)

        # Subtle brightness/contrast adjustments
        if random.random() < self.p:
            brightness_factor = random.uniform(0.95, 1.05)
            img = transforms.functional.adjust_brightness(img, brightness_factor)

        if random.random() < self.p:
            contrast_factor = random.uniform(0.95, 1.05)
            img = transforms.functional.adjust_contrast(img, contrast_factor)

        return img


def calculate_brain_tumor_normalization_stats(train_data):
    """
    Calculate mean and std across all brain tumor images in the dataset.

    Args:
        train_data (list): List of image tensors from training set

    Returns:
        tuple: (means, stds) for each channel
    """
    all_images = torch.stack(train_data)
    means = torch.mean(all_images, dim=[0, 2, 3])
    stds = torch.std(all_images, dim=[0, 2, 3])

    return means.tolist(), stds.tolist()


def load_brain_tumor_dataset(data_path="datasets", seed=42):
    """
    Load and preprocess the brain tumor dataset.

    Args:
        data_path (str): Path to the dataset directory
        seed (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing dataset splits and metadata
    """
    dataset_path = os.path.join(data_path, "brain_tumor")

    # Load preprocessed data from the CSV created by preprocess_brain_tumor.py
    csv_path = os.path.join(dataset_path, "dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

    images = []
    labels = []

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Load and process each image
    for _, row in df.iterrows():
        try:
            # Load image
            img_path = row["image_path"]
            img = Image.open(img_path).convert("RGB")

            # Resize to standard size (224x224)
            img = transforms.Resize((224, 224))(transforms.ToTensor()(img))

            images.append(img)
            labels.append(row["label"])

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

    if not images:
        raise ValueError("No valid images were loaded")

    # Convert labels to numpy array
    labels = np.array(labels)

    # Split into train+val and test sets (80-20)
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    print(f"\n> CV Fold {cv_outer_fold}: Dataset split (train+val/test): {len(train_val_data)}/{len(test_data)} in a 80%/20% split")

    # Calculate class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")

    return {
        "train_val_data": train_val_data,
        "train_val_labels": train_val_labels,
        "test_data": test_data,
        "test_labels": test_labels,
        "num_classes": len(unique_labels),
    }


def get_brain_tumor_kfold_loaders(
    data,
    labels,
    cv_inner_folds,
    batch_size,
    num_workers,
    fold_idx,
    normalization_stats=None,
    augmentation_type="medical",
):
    """
    Create data loaders for k-fold cross validation of brain tumor dataset.

    Args:
        data (list): Combined training and validation data
        labels (numpy.ndarray): Combined training and validation labels
        cv_inner_folds (int): Number of folds for cross-validation
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        fold_idx (int): Current fold index
        normalization_stats (dict, optional): Pre-computed normalization statistics
        augmentation_type (str): Type of augmentation to use

    Returns:
        tuple: (train_loader, val_loader) for the current fold
    """
    # Create k-fold splitter
    kfold = StratifiedKFold(n_splits=cv_inner_folds, shuffle=True, random_state=42)

    # Get indices for current fold
    indices = np.arange(len(data))
    for i, (train_idx, val_idx) in enumerate(kfold.split(indices, labels)):
        if i == fold_idx:
            break

    # Split data for current fold
    train_data = [data[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_data = [data[i] for i in val_idx]
    val_labels = labels[val_idx]

    # Calculate normalization stats from training data if not provided
    if normalization_stats is None:
        means, stds = calculate_brain_tumor_normalization_stats(train_data)
        normalization_stats = {"mean": means, "std": stds}

    # Create datasets
    train_dataset = BrainTumorDataset(
        train_data,
        train_labels,
        normalization_stats=normalization_stats,
        augmentation_type=augmentation_type,
        is_training=True,
    )

    val_dataset = BrainTumorDataset(
        val_data,
        val_labels,
        normalization_stats=normalization_stats,
        augmentation_type=None,
        is_training=False,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_max_batch_size(pipeline_space):
    batch_size = pipeline_space.get("batch_size", None)
    if batch_size is None:
        return 32
    return batch_size.upper


def preprocess_raw_brain_tumor_dataset(dataset_path, output_path):
    """
    Processes the raw brain tumor dataset and creates a CSV file with image paths and labels.

    Args:
        dataset_path (str): Path to original dataset with 'yes' and 'no' folders
        output_path (str): Path to output directory

    Returns:
        pd.DataFrame: DataFrame containing image paths and labels
    """
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Initialize lists for DataFrame
    image_paths = []
    labels = []

    # Process each class
    for class_name in ["no", "yes"]:
        class_path = os.path.join(dataset_path, class_name)
        label = 0 if class_name == "no" else 1

        # Check if the directory exists
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found!")
            continue

        # Process all images in the class
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                # Copy image to output directory
                src_path = os.path.join(class_path, img_name)
                dst_path = os.path.join(output_path, f"{class_name}_{img_name}")
                shutil.copy2(src_path, dst_path)

                # Store path and label
                image_paths.append(dst_path)
                labels.append(label)

    # Create DataFrame
    df = pd.DataFrame({"image_path": image_paths, "label": labels})

    # Save DataFrame to CSV
    csv_path = os.path.join(output_path, "dataset.csv")
    df.to_csv(csv_path, index=False)

    print(f"Processing completed. Dataset info:")
    print(f"Total images: {len(df)}")
    print(f"No tumor images: {len(df[df['label'] == 0])}")
    print(f"Tumor images: {len(df[df['label'] == 1])}")
    print(f"CSV file saved to: {csv_path}")

    return df


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="experimental_setting.yaml",
)
def preprocess_and_cache_brain_tumor_datasets(experimental_setting: DictConfig) -> None:
    """
    Preprocess and cache brain tumor datasets for faster experiment initialization.

    Args:
        experimental_setting (DictConfig): Hydra configuration object
    """
    print("\nPreprocessing brain tumor datasets...")

    # First, process the raw dataset
    raw_dataset_path = os.path.join(experimental_setting.data.path, "brain_mri")
    processed_dataset_path = os.path.join(experimental_setting.data.path, "brain_tumor")

    if not os.path.exists(os.path.join(processed_dataset_path, "dataset.csv")):
        print("Processing raw dataset...")
        preprocess_raw_brain_tumor_dataset(raw_dataset_path, processed_dataset_path)
    else:
        print("Raw dataset already processed, skipping...")

    # Create cache directory
    cache_dir = Path(experimental_setting.data.path) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache filename
    pipeline_space = yaml_to_neps_pipeline_space(experimental_setting.pipeline_space)
    cache_file = cache_dir / f"brain_tumor_bs{get_max_batch_size(pipeline_space)}.pkl"

    if cache_file.exists():
        print(f"Cache file already exists at {cache_file}")
        print("Delete it manually if you want to regenerate the cache.")
        return

    # Load raw dataset
    print("Loading brain tumor dataset...")
    dataset_dict = load_brain_tumor_dataset(
        data_path=experimental_setting.data.path, seed=experimental_setting.seed
    )

    # Calculate normalization statistics from training data only
    print("Calculating dataset-specific normalization statistics...")
    means, stds = calculate_brain_tumor_normalization_stats(
        dataset_dict["train_val_data"]
    )
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
        "normalization_stats",
    ]
    missing_keys = [key for key in required_keys if key not in dataset_dict]
    if missing_keys:
        raise KeyError(f"Dataset dictionary missing required keys: {missing_keys}")

    # Cache the complete dataset dictionary
    print(f"\nSaving cache to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(dataset_dict, f)

    print("\nPreprocessing completed!")
    print(
        f"Brain tumor dataset preprocessed and cached with {dataset_dict['num_classes']} classes"
    )
    print(f"Dataset-specific normalization values have been calculated and cached.")


if __name__ == "__main__":
    preprocess_and_cache_brain_tumor_datasets()
