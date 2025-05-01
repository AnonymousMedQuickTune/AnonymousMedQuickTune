import os
import numpy as np
from sklearn.model_selection import train_test_split


# global variables

IMAGE_NAME = "/image.nii.gz"
SEGMENTATION_NAME = "/segmentation.nii.gz"
MODALITY = "MRI"

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

    images, segmentations, csv_path = get_paths(data_path, name)

    # Load labels
    labels_csv = pd.read_csv(csv_path)
    labels = labels_csv['Diagnosis_binary'].to_numpy()

    # Recheck class distribution after filtering
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution after filtering: {dict(zip(unique_labels, counts))}")

    # Split into train+val and test (80-20)
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    print(f"\nDataset split (train+val/test): {len(train_val_data)}/{len(test_data)}")

    return {
        "train_val_data": train_val_data,
        "train_val_labels": train_val_labels,
        "test_data": test_data,
        "test_labels": test_labels,
        "num_classes": len(unique_labels),
    }
    
def get_paths(data_path, name):
    full_path = os.path.join(data_path, name)
    directory_names = sorted(os.listdir(full_path), key=natural_key)

    image_name = configuration.IMAGE_NAME
    segmentation_name = configuration.SEGMENTATION_NAME

    images_path = [os.path.join(full_path, d, image_name) for d in directory_names]
    segmentations_path = [os.path.join(full_path, d, segmentation_name) for d in directory_names]

    csv_path = os.path.join(full_path, "dataset.csv")

    return images_path, segmentations_path, csv_path

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def cache_datastes(config: DictConfig) -> None:
    """
    Preprocess and cache brain tumor datasets for faster experiment initialization.

    Args:
        config (DictConfig): Hydra configuration object
    """
    print("\nPreprocessing dataset...")


    # First, process the raw dataset
    raw_dataset_path = os.path.join(config.data.path, "Lipo/raw")
    processed_dataset_path = os.path.join(config.data.path, "Lipo/cache")

    if not os.path.exists(os.path.join(raw_dataset_path + "cache" + voxel_key)):
        print("Processing raw dataset...")
        preprocess_dataset(raw_dataset_path, processed_dataset_path, voxel_key)
    else:
        print("Raw dataset already processed, skipping...")


def preprocess_dataset(dataset_path, output_path):

    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Here is all the catched value of the paths for the specific voxel size. 

    X_train, y_train, _, _, seg_train, _ = prepare_data()

    # Combine images and labels into a list of dictionaries
    train_data_images = [{"index": idx, "image": img, "label": label} 
                    for idx, (img, label) in enumerate(zip(X_train, y_train))]
    
    train_data_segmentations = [{"index": idx, "seg": seg} 
                            for idx, seg in enumerate(seg_train)]

    # Prepare train and validation sets
    train_data = [train_data_images[i] for i in train_idx]
    valid_data = [train_data_images[i] for i in val_idx]

    # Segmentation
    train_seg = [train_data_segmentations[i] for i in train_idx]
    valid_seg = [train_data_segmentations[i] for i in val_idx]

    stats = compute_voxel_sizes()

    median_spacing = stats["median_voxel"]
    mean_spacing = stats["mean_voxel"]
    isotropic_spacing = stats["isotropic_voxel"]
    isovolumetric_spacing = stats["isotropic_volume_voxel"]
    
    # Define target spacing values for each voxel option
    target_spacing_options = {
        "Median": median_spacing,
        "Mean": mean_spacing,
        "Isotropic": isotropic_spacing,
        "IsoVolumetric": isovolumetric_spacing
    }

    default_spacing = median_spacing

    # Get the corresponding target spacing based on the selected voxel option
    target_spacing = target_spacing_options.get(selected_voxel, default_spacing)

    # Print or use the target spacing
    print("Selected Target Spacing:", target_spacing)

    # First preprocess part:
    train_set_images = Dataset(train_data, transform=PreImgTransform(target_spacing))
    train_set_segmentations = Dataset(train_seg, transform=PreSegTransform(target_spacing))
    valid_set_images = Dataset(valid_data, transform=PreImgTransform(target_spacing))
    valid_set_segmentations = Dataset(valid_seg, transform=PreSegTransform(target_spacing))

    # First for training data
    train_roi_start, train_roi_end = cropping_padding(train_set_images, train_set_segmentations)
    
    train_set = Dataset(train_data, transform=FullTransform(target_spacing, train_roi_start, train_roi_end))

    # Loop through dataset and check shapes
    print(len(train_set_images))

    # Cropping for validation data
    valid_roi_start, valid_roi_end = cropping_padding(valid_set_images, valid_set_segmentations)
    valid_set = Dataset(valid_data, transform=ValidTransform(target_spacing, valid_roi_start, valid_roi_end))

    with open(train_path, "wb") as f:
        pickle.dump(train_set, f)
    with open(val_path, "wb") as f:
        pickle.dump(valid_set, f)

    return train_set, valid_set

def get_dataloaders(
    data,
    labels,
    k_folds,
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
        k_folds (int): Number of folds for cross-validation
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        fold_idx (int): Current fold index
        normalization_stats (dict, optional): Pre-computed normalization statistics
        augmentation_type (str): Type of augmentation to use

    Returns:
        tuple: (train_loader, val_loader) for the current fold
    """

    # Create k-fold splitter
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

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

    # here should come everything that has to be about the train set

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