import numpy as np
from sklearn.model_selection import train_test_split

def load_3d_dataset(name, data_path="datasets", seed=42):
    """
    Load and preprocess a medical image dataset.

    Args:
        name (str): Name of the dataset to load ('lipo', 'desmoid', 'gist')
        data_path (str): Base path to the datasets directory. Defaults to 'datasets'

    Returns:
        dict: Dictionary containing dataset splits and metadata
    """

    # TODO: Implement 3D dataset loading

    images = None  # placeholder
    labels = None  # placeholder

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