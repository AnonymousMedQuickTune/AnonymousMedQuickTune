import os
import torch
import numpy as np
import re
import json
import pickle
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.classification_2d.models_2d import get_2d_model
from src.classification_2d.preprocess_data_2d import BrainTumorDataset, get_max_batch_size, load_brain_tumor_dataset
from src.classification_3d.models_3d import get_3d_model
from src.classification_3d.preprocess_data_3d import calculate_voxel_size_from_images
from src.classification_3d.preprocess_data_3d import DataTransform
from src.classification_3d.utils.dataset_info import extract_spatial_size
from src.utils.common_utils import set_seed, set_reproducibility_env_vars, print_reproducibility_info, yaml_to_neps_pipeline_space
from src.utils.model_lifecycle_utils import evaluate_model
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from monai.data import Dataset

def load_normalization_stats_from_fold(pipeline_directory, fold_idx):
    """
    Load normalization statistics from the normalization_stats.txt file of a specific inner CV fold.
    
    Args:
        pipeline_directory (str): Directory containing the trained model checkpoints
        fold_idx (int): Inner CV fold index (0-based)
    
    Returns:
        dict or None: Dictionary containing 'mean' and 'std' keys with list values, or None if file not found
    """
    # Construct path to normalization_stats.txt file
    normalization_stats_file = Path(pipeline_directory) / f"cv_inner_fold_{fold_idx}" / "normalization_stats.txt"
    
    if not normalization_stats_file.exists():
        print(f"Warning: Normalization stats file not found at {normalization_stats_file}")
        return None
    
    try:
        with open(normalization_stats_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse mean and std values using regex
        # Look for patterns like "Mean: [6.358785321936011e-05]" and "Std:  [0.000162713389727287]"
        mean_match = re.search(r"Mean:\s*\[([^\]]+)\]", content)
        std_match = re.search(r"Std:\s*\[([^\]]+)\]", content)
        
        # Parse percentiles if available (format: "Percentiles (for clipping): [lower, upper]" or "Percentiles (for clipping): None")
        percentiles_match = re.search(r"Percentiles \(for clipping\):\s*(?:\[([^\]]+)\]|None)", content)
        
        if mean_match and std_match:
            # Extract the numeric values and convert to float
            mean_value = float(mean_match.group(1))
            std_value = float(std_match.group(1))
            
            normalization_stats = {
                "mean": [mean_value],
                "std": [std_value]
            }
            
            # Parse percentiles if available
            if percentiles_match:
                if percentiles_match.group(1) is None:
                    # No percentiles (e.g., from AutoNorm) - matched "None"
                    normalization_stats["percentiles"] = None
                else:
                    # Parse percentile values: "[lower, upper]" format
                    perc_str = percentiles_match.group(1)
                    perc_values = [float(x.strip()) for x in perc_str.split(",")]
                    normalization_stats["percentiles"] = perc_values
                    print(f"Loaded percentiles for clipping: [{perc_values[0]:.2f}, {perc_values[1]:.2f}]")
            else:
                # No percentiles line found (backward compatibility with old files)
                normalization_stats["percentiles"] = None
            
            print(f"\nLoaded normalization stats for fold {fold_idx}: mean={mean_value:.6e}, std={std_value:.6e}")
            return normalization_stats
        else:
            print(f"Warning: Could not parse normalization stats from {normalization_stats_file}")
            return None
            
    except Exception as e:
        print(f"Error reading normalization stats file {normalization_stats_file}: {e}")
        return None

def calculate_metrics_from_probabilities(probabilities, ground_truth_targets, num_classes):
    """
    Calculate comprehensive metrics from probability predictions.
    
    This is a shared function that calculates all metrics (macro/micro averaged) 
    from probability predictions and ground truth labels.
    
    Args:
        probabilities (np.ndarray): Probability predictions
            Shape: (num_samples, num_classes)
        ground_truth_targets (np.ndarray): Ground truth labels
            Shape: (num_samples,) - single array with true class labels
        num_classes (int): Number of classes in the dataset
    
    Returns:
        dict: Comprehensive metrics dictionary with macro/micro averaged metrics
    """
    # BASIC CLASSIFICATION METRICS
    # ------------------------------------------------------------------------------------------------
    # Convert probabilities to predictions
    predictions = np.argmax(probabilities, axis=1)
    
    # Calculate accuracy: fraction of correct predictions
    accuracy = np.mean(predictions == ground_truth_targets)
    
    # PER-CLASS METRICS CALCULATION
    # ------------------------------------------------------------------------------------------------
    # Calculate precision, recall, F1 for each class individually
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        ground_truth_targets,
        predictions,
        average=None,         # Calculate metrics for each class separately
        zero_division=0,      # Handle division by zero gracefully
    )
    
    # CONFUSION MATRIX
    # ------------------------------------------------------------------------------------------------
    # Confusion matrix shows prediction vs. actual class distribution
    conf_matrix = confusion_matrix(ground_truth_targets, predictions)
    
    # AUC CALCULATION (PROBABILITY-BASED)
    # ------------------------------------------------------------------------------------------------
    # AUC uses probabilities, not hard predictions, so it benefits from ensemble averaging
    if num_classes == 2:
        # Binary classification: use probability of positive class
        auc_macro = roc_auc_score(ground_truth_targets, probabilities[:, 1])
        auc_micro = auc_macro  # For binary classification, macro = micro
    else:
        # Multiclass: Calculate both macro and micro averaged AUC
        # Macro: equal weight to all classes (good for imbalanced datasets)
        auc_macro = roc_auc_score(ground_truth_targets, probabilities, multi_class="ovr", average="macro")
        # Micro: weighted by class frequency (gives more weight to frequent classes)
        auc_micro = roc_auc_score(ground_truth_targets, probabilities, multi_class="ovr", average="micro")
    
    # MACRO AND MICRO-AVERAGED METRICS
    # ------------------------------------------------------------------------------------------------
    # Macro averaging treats all classes equally (good for imbalanced datasets)
    # Micro averaging weights by class frequency (gives more weight to frequent classes)
    precision_macro = np.mean(precision_per_class)
    recall_macro = np.mean(recall_per_class)
    f1_macro = np.mean(f1_per_class)
    
    # Calculate micro-averaged metrics (weighted by class frequency)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        ground_truth_targets, predictions, average="micro", zero_division=0
    )

    # CREATE COMPREHENSIVE METRICS DICTIONARY
    # ------------------------------------------------------------------------------------------------
    # Structure: scalar metrics for summary + detailed per-class breakdown
    metrics = {
        # Basic metrics
        "accuracy": float(accuracy * 100),
        
        # AUC metrics (macro and micro)
        "auc_macro": float(auc_macro * 100),
        "auc_micro": float(auc_micro * 100),
        
        # Macro-averaged metrics (equal weight to all classes)
        "precision_macro": float(precision_macro * 100),
        "recall_macro": float(recall_macro * 100),
        "f1_macro": float(f1_macro * 100),
        
        # Micro-averaged metrics (weighted by class frequency)
        "precision_micro": float(precision_micro * 100),
        "recall_micro": float(recall_micro * 100),
        "f1_micro": float(f1_micro * 100),
        
        # Detailed breakdowns
        "confusion_matrix": conf_matrix.tolist(),  # Convert numpy array to list for JSON serialization
        "per_class": {
            "precision": (precision_per_class * 100).tolist(),
            "recall": (recall_per_class * 100).tolist(),
            "f1": (f1_per_class * 100).tolist(),
        },
    }
    
    return metrics

def calculate_ensemble_metrics(folds_probabilities, ground_truth_targets, num_classes):
    """
    Calculate comprehensive ensemble metrics by averaging softmax probabilities across cross-validation folds.
    
    This implements cross-validation ensemble learning (bagging) where:
    1. Each fold's model predicts on the entire test set
    2. Softmax probabilities are averaged across folds (not hard predictions!)
    3. Final predictions are derived from averaged probabilities
    4. Both macro and micro averaged metrics are calculated
    
    Args:
        folds_probabilities (list): List of probability arrays, one per fold
            Shape: [fold_0_probs, fold_1_probs, ...]
            Each fold_probs has shape (num_samples, num_classes)
        ground_truth_targets (np.ndarray): Ground truth labels for the test set (same for all folds)
            Shape: (num_samples,) - single array with true class labels
        num_classes (int): Number of classes in the dataset
    
    Returns:
        dict: Comprehensive ensemble metrics dictionary containing:
            - accuracy: Overall classification accuracy
            - auc_macro/auc_micro: AUC with macro/micro averaging
            - precision_macro/precision_micro: Precision with macro/micro averaging
            - recall_macro/recall_micro: Recall with macro/micro averaging
            - f1_macro/f1_micro: F1-score with macro/micro averaging
            - confusion_matrix: Confusion matrix as nested list
            - per_class: Dictionary with per-class precision, recall, F1
    
    Note:
        - Macro averaging: Equal weight to all classes (good for imbalanced datasets)
        - Micro averaging: Weighted by class frequency (gives more weight to frequent classes)
        - For binary classification, macro AUC = micro AUC
    """
    
    # ENSEMBLE PROBABILITY AVERAGING
    # ------------------------------------------------------------------------------------------------
    # Stack all fold probabilities into a 3D array
    # shape: (num_folds, num_samples, num_classes)
    stacked_probabilities = np.stack(folds_probabilities, axis=0)  

    # Then average across folds to get ensemble probabilities
    # shape: (num_samples, num_classes)
    avg_probabilities = np.mean(stacked_probabilities, axis=0)  
    
    print(f"\n=== Calculating Final Metrics from Ensemble Predictions ===")
    print(f"Total test samples: {len(ground_truth_targets)}")
    print(f"Folds ensembled: {len(folds_probabilities)}")
    print(f"Ensemble probability shape: {avg_probabilities.shape}")
    
    # CALCULATE METRICS USING SHARED FUNCTION
    # ------------------------------------------------------------------------------------------------
    # Use the shared metrics calculation function
    ensemble_metrics = calculate_metrics_from_probabilities(avg_probabilities, ground_truth_targets, num_classes)
    
    # PRINT ENSEMBLE RESULTS
    # ------------------------------------------------------------------------------------------------
    print(f"\n=== Final Test Set Results (Ensemble Predictions) ===")
    print(f"Accuracy: {ensemble_metrics['accuracy']:.2f}%")
    print(f"\nAUC metrics:")
    print(f"  Macro-averaged AUC: {ensemble_metrics['auc_macro']:.2f}%")
    print(f"  Micro-averaged AUC: {ensemble_metrics['auc_micro']:.2f}%")
    print(f"\nMacro-averaged metrics (equal weight to all classes):")
    print(f"  Precision: {ensemble_metrics['precision_macro']:.2f}%")
    print(f"  Recall: {ensemble_metrics['recall_macro']:.2f}%")
    print(f"  F1: {ensemble_metrics['f1_macro']:.2f}%")
    print(f"\nMicro-averaged metrics (weighted by class frequency):")
    print(f"  Precision: {ensemble_metrics['precision_micro']:.2f}%")
    print(f"  Recall: {ensemble_metrics['recall_micro']:.2f}%")
    print(f"  F1: {ensemble_metrics['f1_micro']:.2f}%")
    print(f"{'='*80}\n")
    
    return ensemble_metrics

def calculate_per_fold_metrics(folds_probabilities, ground_truth_targets, num_classes):
    """
    Calculate comprehensive per-fold metrics for detailed cross-validation analysis.
    
    This function evaluates each fold individually to provide insights into:
    - Individual fold performance variations
    - Per-fold metric distributions
    - Detailed breakdown of ensemble components
    
    Args:
        folds_probabilities (list): List of probability arrays, one per fold
            Shape: [fold_0_probs, fold_1_probs, ...]
            Each fold_probs has shape (num_samples, num_classes)
        ground_truth_targets (np.ndarray): Ground truth labels for the test set (same for all folds)
            Shape: (num_samples,) - single array with true class labels
        num_classes (int): Number of classes in the dataset
    
    Returns:
        list: List of dictionaries, one per fold, each containing:
            - fold_index: Index of the fold (0-based)
            - metrics: Dictionary with comprehensive fold metrics including:
                - accuracy: Classification accuracy for this fold
                - auc_macro/auc_micro: AUC with macro/micro averaging
                - precision_macro/precision_micro: Precision with macro/micro averaging
                - recall_macro/recall_micro: Recall with macro/micro averaging
                - f1_macro/f1_micro: F1-score with macro/micro averaging
                - confusion_matrix: Confusion matrix as nested list
                - per_class: Dictionary with per-class precision, recall, F1
    
    Note:
        - Each fold uses the same test set but different trained models
        - Per-fold metrics help identify fold-specific performance patterns
        - Useful for analyzing ensemble stability and fold consistency
    """
    per_fold_summaries = []
    
    # PER-FOLD METRICS CALCULATION
    # ------------------------------------------------------------------------------------------------
    # Calculate metrics for each fold individually
    for fold_idx, fold_probs in enumerate(folds_probabilities):
        # Use the shared metrics calculation function
        fold_metrics = calculate_metrics_from_probabilities(fold_probs, ground_truth_targets, num_classes)
        
        # Build per-fold dictionary
        fold_summary = {
            "fold_index": fold_idx,
            "metrics": fold_metrics,
        }
        
        per_fold_summaries.append(fold_summary)
        
        # Print fold results
        print(f"Fold {fold_idx + 1} Test Results:")
        print(f"Accuracy: {fold_metrics['accuracy']:.2f}%")
        print(f"AUC metrics:")
        print(f"  Macro-averaged AUC: {fold_metrics['auc_macro']:.2f}%")
        print(f"  Micro-averaged AUC: {fold_metrics['auc_micro']:.2f}%")
        print(f"Macro-averaged metrics (equal weight to all classes):")
        print(f"  Precision: {fold_metrics['precision_macro']:.2f}%")
        print(f"  Recall: {fold_metrics['recall_macro']:.2f}%")
        print(f"  F1: {fold_metrics['f1_macro']:.2f}%")
        print(f"Micro-averaged metrics (weighted by class frequency):")
        print(f"  Precision: {fold_metrics['precision_micro']:.2f}%")
        print(f"  Recall: {fold_metrics['recall_micro']:.2f}%")
        print(f"  F1: {fold_metrics['f1_micro']:.2f}%\n")
    
    return per_fold_summaries


def evaluate_fold(fold, test_loader, model, experimental_setting, hyperparameters, pipeline_directory):
    """
    Evaluate a single fold's trained model on the test set and return prediction probabilities.
    
    This function loads a trained model checkpoint for a specific fold and evaluates it on the test set.
    It returns softmax probabilities (not hard predictions) to enable ensemble averaging.
    
    Args:
        fold (int): Fold index (0-based) for which to evaluate the model
        test_loader (DataLoader): PyTorch DataLoader containing test data
        model (nn.Module): Pre-initialized model (architecture is the same for all folds)
        experimental_setting (DictConfig): Hydra configuration object with experiment settings
        hyperparameters (dict): Hyperparameters used for training the model
        pipeline_directory (str): Directory containing the trained model checkpoints
    
    Returns:
        tuple: (fold_probabilities, fold_targets) containing:
            - fold_probabilities (np.ndarray): Softmax probabilities for test samples
                Shape: (num_samples, num_classes)
            - fold_targets (list): Ground truth labels for test samples
                Shape: (num_samples,) - list of true class labels
    
    Note:
        - Uses the best model checkpoint saved during training
        - Returns probabilities (not predictions) for ensemble averaging
        - Ground truth targets are the same for all folds (same test set)
        - Model weights are loaded from fold-specific checkpoints
        - Works for both NePS and QuickTune frameworks
    """
    # Set device for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MODEL CHECKPOINT LOADING
    # ------------------------------------------------------------------------------------------------
    # Load the trained model checkpoint for this fold
    if experimental_setting.training.no_validation or not experimental_setting.training.early_stopping:
        model_checkpoint = "model_latest_checkpoint.pth"
    else:
        model_checkpoint = "best_model_checkpoint.pth"

    checkpoint_path = (
        Path(pipeline_directory)
        / f"cv_inner_fold_{fold}"
        / model_checkpoint
    )
    
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        return None, None
    
    # Add safe globals for NumPy objects that might be in the checkpoint
    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar, 
        np.dtype, 
        np.dtypes.Float64DType,
        np.dtypes.StrDType,  # For string dtypes that might be in checkpoints
    ])
    
    # Load the model checkpoint with minimal fallback
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
    except Exception:
        # Fallback for checkpoints with unsupported NumPy types
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # LOSS FUNCTION SETUP
    # ------------------------------------------------------------------------------------------------
    # Setup loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=hyperparameters.get("label_smoothing", 0.0)
    )
    
    # MODEL EVALUATION ON TEST DATA
    # ------------------------------------------------------------------------------------------------
    # Collect per-sample probabilities and targets for this fold (no shuffling → consistent order)
    fold_probabilities = []
    fold_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                # Batch is a dict for 3D datasets
                inputs = batch.get("image")
                targets = batch.get("label")
            else:
                # Batch is a tuple for 2D datasets
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            fold_probabilities.extend(probabilities.detach().cpu().numpy())
            fold_targets.extend(targets.cpu().numpy())
    
    return fold_probabilities, fold_targets

# TODO @Diane: double-check this function
def evaluate_config_on_validation_set_ensemble(
    pipeline_directory,
    experimental_setting,
    dataset,
    spatial_size,
    num_classes,
    hyperparameters,
    cv_inner_folds_splits,
    cv_inner_folds_repeats,
    total_inner_folds,
    seed,
    framework="neps"
):
    """
    Evaluate a trained configuration on the validation set using ensemble predictions.
    
    This function implements cross-validation ensemble learning for validation data where:
    1. For each sample, only models that did NOT train on that sample are used
    2. Softmax probabilities are averaged across these models (not hard predictions!)
    3. Final predictions are derived from averaged probabilities
    4. Comprehensive metrics are calculated
    
    This is methodologically correct: each model only evaluates samples it hasn't seen during training.
    
    Args:
        pipeline_directory (str): Directory containing the trained model checkpoints
        experimental_setting (DictConfig): Hydra configuration object with experiment settings
        dataset (dict): Dictionary containing train_val_images and train_val_labels (already selected based on voxel_calculation)
        spatial_size (tuple): Spatial size tuple for model initialization (already calculated)
        num_classes (int): Number of classes in the classification task
        hyperparameters (dict): Hyperparameters used for training the model
        cv_inner_folds_splits (int): Number of splits per repetition
        cv_inner_folds_repeats (int): Number of repetitions
        total_inner_folds (int): Total number of folds (repeats * splits)
        seed (int): Random seed used for generating splits
        framework (str): Framework being used ("neps" or "quicktune")
    
    Returns:
        dict: Comprehensive validation ensemble metrics dictionary containing:
            - accuracy: Overall classification accuracy
            - auc: AUC score
            - precision: Precision (macro-averaged)
            - recall: Recall (macro-averaged)
            - f1: F1-score (macro-averaged)
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING CONFIG ON VALIDATION SET (ENSEMBLE)")
    print(f"{'='*80}\n")
    
    # CRITICAL: Set environment variables for reproducibility FIRST
    # This ensures reproducibility even if this function is called directly
    set_reproducibility_env_vars()
    
    # Set seed for reproducibility (this also sets PYTHONHASHSEED and deterministic flags)
    set_seed(experimental_setting.seed)
    
    # Print reproducibility info for debugging AFTER setting seed (helps identify differences between environments)
    # Note: This is only printed once per function call, not for every fold
    print_reproducibility_info()
    
    # Load validation data (train_val_images and train_val_labels)
    if experimental_setting.data.dimensionality.lower() != "3d":
        raise NotImplementedError("2D evaluation is not supported for validation ensemble yet.")
    
    # MODEL INITIALIZATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if framework == "quicktune":
        model_type = hyperparameters["model"]
    elif framework == "neps":
        model_type = experimental_setting.model.type
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    # Initialize model and move it to the appropriate device
    model_config = {"type": model_type, "task": experimental_setting.model.task, "num_classes": num_classes}
    model = get_3d_model(
        model_config=model_config,
        hyperparameters=hyperparameters,
        developer_mode=experimental_setting.developer_mode,
        spatial_size=spatial_size,
        is_medmnist=dataset.get("is_medmnist", False)
    ).to(device)
    
    # STEP 1: Load fold splits that were saved during training
    print("Loading fold splits from training...")
    train_val_images = dataset["train_val_images"]
    train_val_labels = np.array(dataset["train_val_labels"])
    
    # Load splits from file (saved during training in get_kfold_dataloaders)
    splits_file = os.path.join(pipeline_directory, "inner_cv_splits.pkl")
    
    if not os.path.exists(splits_file):
        raise FileNotFoundError(
            f"Inner CV splits file not found at {splits_file}. "
            "This file should be created during training. Please ensure training has completed at least one fold."
        )
    
    with open(splits_file, "rb") as f:
        splits_data = pickle.load(f)
    
    all_splits = splits_data["splits"]
    print(f"Loaded {len(all_splits)} fold splits from training")
    print(f"Split parameters: n_repeats={splits_data['n_repeats']}, n_splits={splits_data['n_splits']}, seed={splits_data['seed']}")
    
    # Verify that the number of samples matches
    if splits_data["total_samples"] != len(train_val_images):
        raise ValueError(
            f"Mismatch in number of samples: splits file has {splits_data['total_samples']} samples, "
            f"but dataset has {len(train_val_images)} samples. This might indicate a different dataset or data preprocessing."
        )
    
    # Store which samples are in which validation sets
    sample_to_val_folds = {}  # sample_idx -> list of fold indices where this sample is in validation set
    sample_to_train_folds = {}  # sample_idx -> list of fold indices where this sample is in training set
    
    for fold_idx, (train_idx, val_idx) in enumerate(all_splits):
        # Mark which samples are in validation set for this fold
        for val_sample_idx in val_idx:
            if val_sample_idx not in sample_to_val_folds:
                sample_to_val_folds[val_sample_idx] = []
            sample_to_val_folds[val_sample_idx].append(fold_idx)
        
        # Mark which samples are in training set for this fold
        for train_sample_idx in train_idx:
            if train_sample_idx not in sample_to_train_folds:
                sample_to_train_folds[train_sample_idx] = []
            sample_to_train_folds[train_sample_idx].append(fold_idx)
    
    print(f"Samples in validation sets: {len(sample_to_val_folds)}")
    print(f"Samples in training sets: {len(sample_to_train_folds)}")
    
    # STEP 2: Evaluate each fold's model on samples it didn't train on
    # For each sample, collect probabilities from models that didn't train on it
    all_sample_probabilities = {}  # sample_idx -> list of probability arrays from different models
    all_sample_targets = {}  # sample_idx -> target label
    
    # Prepare data for evaluation
    val_data = [{"index": idx, "image": img, "label": label} 
                for idx, (img, label) in enumerate(zip(train_val_images, train_val_labels))]
    
    for fold in range(total_inner_folds):
        print(f"\n=== Evaluating Fold {fold + 1}/{total_inner_folds} ===")
        
        # Load normalization stats from this fold
        normalization_stats = load_normalization_stats_from_fold(pipeline_directory, fold)
        
        # For MRI datasets (lipo, desmoid, liver), normalization_stats is None because
        # normalization is done per image/patient in preprocessing. This is expected.
        # For CT datasets, normalization_stats should be available from the saved file.
        dataset_name = experimental_setting.data.dataset.lower()
        is_mri_dataset = dataset_name in ["lipo", "desmoid", "liver"]
        
        if normalization_stats is None:
            if is_mri_dataset:
                print(f"Note: No normalization stats file found for fold {fold} (expected for MRI datasets)")
                # normalization_stats remains None, which is correct for MRI datasets
            else:
                print(f"Warning: Normalization stats not found for fold {fold} (CT dataset), skipping...")
                continue
        
        # Create dataset with transforms
        val_dataset = Dataset(
            val_data, 
            transform=DataTransform(
                normalization_stats, 
                developer_mode=experimental_setting.developer_mode, 
                spatial_size=spatial_size, 
                is_training=False, 
                is_medmnist=dataset.get("is_medmnist", False), 
                augmentation_type=experimental_setting.data.augmentation_type
            )
        )
        
        # Worker init function to ensure each worker has a deterministic seed
        # This is critical when num_workers > 0 for reproducibility
        def worker_init_fn(worker_id):
            # Set seed for each worker based on the base seed and worker ID
            # This ensures reproducibility even with multiple workers
            worker_seed = experimental_setting.seed + worker_id
            set_seed(worker_seed)

        # Create data loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=hyperparameters.get(
                "batch_size",
                getattr(experimental_setting.training, "batch_size", 1)
            ),
            shuffle=False,
            num_workers=experimental_setting.data.num_workers,
            pin_memory=False,
            worker_init_fn=worker_init_fn if experimental_setting.data.num_workers > 0 else None,  # Deterministic workers
        )
        
        # Evaluate this fold's model
        fold_probabilities, fold_targets = evaluate_fold(
            fold, val_loader, model, experimental_setting, hyperparameters, pipeline_directory
        )
        
        if fold_probabilities is None or fold_targets is None:
            print(f"Warning: Could not evaluate fold {fold}, skipping...")
            continue
        
        # For each sample, if this model didn't train on it, add its probabilities
        # CRITICAL: sample_idx here is the position in val_data (0, 1, 2, ...), which corresponds
        # to the index in train_val_images. This matches the indices used in sample_to_train_folds
        # which are also based on the position in train_val_images.
        for sample_idx, (prob, target) in enumerate(zip(fold_probabilities, fold_targets)):
            # Verify: sample_idx should be in range [0, len(train_val_images))
            if sample_idx >= len(train_val_images):
                print(f"Warning: sample_idx {sample_idx} >= len(train_val_images) {len(train_val_images)}, skipping...")
                continue
            
            # Check if this model trained on this sample
            # IMPORTANT: A sample can be in BOTH train and val sets across different folds
            # We need to check if this specific fold trained on this sample
            if sample_idx in sample_to_train_folds and fold in sample_to_train_folds[sample_idx]:
                # This model trained on this sample, skip it
                continue
            
            # This model didn't train on this sample, use its prediction
            if sample_idx not in all_sample_probabilities:
                all_sample_probabilities[sample_idx] = []
                all_sample_targets[sample_idx] = target
            
            all_sample_probabilities[sample_idx].append(prob)
    
    # STEP 3: Average probabilities for each sample and calculate metrics
    print(f"\n=== Calculating Ensemble Metrics ===")
    print(f"Total samples with predictions: {len(all_sample_probabilities)}")
    print(f"Total samples in dataset: {len(train_val_images)}")
    
    # Check if we have predictions for all samples
    samples_without_predictions = set(range(len(train_val_images))) - set(all_sample_probabilities.keys())
    if samples_without_predictions:
        print(f"Warning: {len(samples_without_predictions)} samples have no predictions (not in any validation set)")
    
    # Collect ensemble probabilities and targets
    ensemble_probabilities = []
    ensemble_targets = []
    
    for sample_idx in sorted(all_sample_probabilities.keys()):
        sample_probs = all_sample_probabilities[sample_idx]
        if len(sample_probs) == 0:
            print(f"Warning: Sample {sample_idx} has empty probability list, skipping...")
            continue
        
        # Average probabilities across models that didn't train on this sample
        # Convert list of arrays to numpy array first, then average over models (axis=0)
        # sample_probs is a list of arrays, each with shape (num_classes,)
        # After stacking: shape (num_models, num_classes)
        # After mean(axis=0): shape (num_classes,)
        avg_probs = np.mean(np.array(sample_probs), axis=0)
        ensemble_probabilities.append(avg_probs)
        ensemble_targets.append(all_sample_targets[sample_idx])
    
    if len(ensemble_probabilities) == 0:
        raise ValueError("No valid ensemble predictions found. All samples were skipped or had empty probability lists.")
    
    ensemble_probabilities = np.array(ensemble_probabilities)
    ensemble_targets = np.array(ensemble_targets)
    
    print(f"Ensemble predictions shape: {ensemble_probabilities.shape}")
    print(f"Number of models per sample (first 10): {[len(all_sample_probabilities[idx]) for idx in sorted(all_sample_probabilities.keys())[:10]]}")
    
    # Debug: Check distribution of number of models per sample
    models_per_sample = [len(all_sample_probabilities[idx]) for idx in all_sample_probabilities.keys()]
    print(f"Models per sample stats: min={min(models_per_sample)}, max={max(models_per_sample)}, mean={np.mean(models_per_sample):.2f}")
    
    # Debug: Check if samples in validation sets have correct number of models
    # For RepeatedStratifiedKFold with n_repeats=R, n_splits=K:
    # - Each sample is in exactly R validation sets (once per repeat)
    # - Each sample is in (K-1)*R training sets
    # - Each sample should be evaluated by R models (those folds where it's in validation set, 
    #   which are the folds that didn't train on it)
    expected_models_per_sample = cv_inner_folds_repeats
    print(f"Expected models per sample: {expected_models_per_sample} (splits={cv_inner_folds_splits}, repeats={cv_inner_folds_repeats}, total_folds={total_inner_folds})")
    
    # Debug: Verify for a few samples
    print(f"\n=== Debug: Sample-to-Fold Mapping (first 5 samples) ===")
    for sample_idx in range(min(5, len(train_val_images))):
        val_folds = sample_to_val_folds.get(sample_idx, [])
        train_folds = sample_to_train_folds.get(sample_idx, [])
        models_used = len(all_sample_probabilities.get(sample_idx, []))
        print(f"Sample {sample_idx}: in val_folds={val_folds} ({len(val_folds)}), in train_folds={train_folds} ({len(train_folds)}), models_used={models_used}, expected={expected_models_per_sample}")
        if models_used != expected_models_per_sample:
            print(f"  ⚠️  WARNING: Sample {sample_idx} has {models_used} models, expected {expected_models_per_sample}")
    
    # Calculate metrics
    ensemble_metrics = calculate_metrics_from_probabilities(
        ensemble_probabilities, ensemble_targets, num_classes
    )
    
    # Convert to format expected by objective_function_3d
    result = {
        "accuracy": ensemble_metrics["accuracy"],
        "auc": ensemble_metrics["auc_macro"],  # Use macro-averaged AUC
        "precision": ensemble_metrics["precision_macro"],
        "recall": ensemble_metrics["recall_macro"],
        "f1": ensemble_metrics["f1_macro"],
    }
    
    print(f"\n=== Validation Ensemble Results ===")
    print(f"Accuracy: {result['accuracy']:.2f}%")
    print(f"AUC: {result['auc']:.2f}%")
    print(f"Precision: {result['precision']:.2f}%")
    print(f"Recall: {result['recall']:.2f}%")
    print(f"F1: {result['f1']:.2f}%")
    print(f"{'='*80}\n")
    
    return result

def evaluate_config_on_test_set(
    pipeline_directory,
    experimental_setting,
    dataset_dict,
    num_classes,
    hyperparameters,
    cv_outer_fold=1,
    framework="neps"  # "neps" or "quicktune"
):
    """
    Evaluate a trained configuration on the test set using cross-validation ensemble predictions.
    
    This function implements cross-validation ensemble learning (bagging) where:
    1. Each fold's trained model predicts on the entire test set
    2. Softmax probabilities are averaged across folds (not hard predictions!)
    3. Final predictions are derived from averaged probabilities
    4. Comprehensive metrics are calculated including macro/micro averaged precision, recall, F1, and AUC
    
    Args:
        pipeline_directory (str): Directory containing the trained model checkpoints
        experimental_setting (DictConfig): Hydra configuration object with experiment settings
        dataset_dict (dict): Dictionary containing dataset information and metadata
        num_classes (int): Number of classes in the classification task
        hyperparameters (dict): Hyperparameters used for training the model
        cv_outer_fold (int): Current cross-validation fold index (0-based, default: 1)
        framework (str): Framework being used ("neps" or "quicktune"), affects model loading
    
    Returns:
        dict: Comprehensive test metrics dictionary containing:
            - ensemble: Dictionary with ensemble-level metrics (macro/micro averaged)
            - per_fold: List of dictionaries with per-fold metrics and summaries
    
    Note:
        - Test data is normalized using fold-specific statistics from training data
        - Each fold uses its own normalization_stats.json file
        - Ground truth labels are the same for all folds (same test set)
        - Ensemble predictions are created by averaging softmax probabilities across folds
        - Works for both NePS and QuickTune frameworks
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING CONFIG ON TEST SET (CV Fold {cv_outer_fold})")
    print(f"{'='*80}\n")
    
    # CRITICAL: Set environment variables for reproducibility FIRST
    # This ensures reproducibility even if this function is called directly
    set_reproducibility_env_vars()
    
    # Set seed for reproducibility (this also sets PYTHONHASHSEED and deterministic flags)
    set_seed(experimental_setting.seed)
    
    # Print reproducibility info for debugging AFTER setting seed (helps identify differences between environments)
    # Note: This is only printed once per function call, not for every fold
    print_reproducibility_info()

    # TEST DATA LOADING
    # ------------------------------------------------------------------------------------------------
    # Load test data based on dimensionality
    if experimental_setting.data.dimensionality.lower() == "2d":
        raise NotImplementedError("2D evaluation is not supported for config evaluation yet.")
        
    elif experimental_setting.data.dimensionality.lower() == "3d":
        # select the dataset_dict based on the selected voxel calculation
        if "voxel_calculation" in str(experimental_setting.pipeline_space):
            if hyperparameters["voxel_calculation"] == "mean":
                dataset = dataset_dict["dataset_dict_mean"]
                voxel_calculation = "mean"
            elif hyperparameters["voxel_calculation"] == "median":
                dataset = dataset_dict["dataset_dict_median"]
                voxel_calculation = "median"
            elif hyperparameters["voxel_calculation"] == "isotropic":
                dataset = dataset_dict["dataset_dict_isotropic"]
                voxel_calculation = "isotropic"
            elif hyperparameters["voxel_calculation"] == "volumetric_isotropic":
                dataset = dataset_dict["dataset_dict_volumetric_isotropic"]
                voxel_calculation = "volumetric_isotropic"
            else:
                raise ValueError(f"Invalid voxel calculation method: {hyperparameters['voxel_calculation']}")
            voxel_size = dataset["voxel_size"]
        else:
            # Use dataset_dict with median voxel calculation
            if experimental_setting.run_mode == "Baseline":
                dataset = dataset_dict
            else:
                dataset = dataset_dict["dataset_dict_median"]
            voxel_size = dataset["voxel_size"]
            voxel_calculation = "median"
        
        # Get image size based on developer mode, model type and voxel size
        spatial_size = extract_spatial_size(
            experimental_setting.model.type, 
            voxel_calculation, 
            experimental_setting.data.dataset, 
            experimental_setting.developer_mode,
            data_path=experimental_setting.data.path,
            is_medmnist=dataset.get("is_medmnist", False)
        )

        # MODEL INITIALIZATION
        # ------------------------------------------------------------------------------------------------
        # Initialize the model once (all folds use the same architecture, only weights differ)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get model type from hyperparameters or experimental_setting
        if framework == "quicktune":
            model_type = hyperparameters["model"]  # For QuickTune
            print(f"\nQuickTune selected model: {model_type}\n")
        elif framework == "neps":
            model_type = experimental_setting.model.type  # For NePS
            print(f"\nNePS selected model: {model_type}\n")
        else:
            raise ValueError(f"Unsupported framework: {framework}. Must be either 'quicktune' or 'neps'.")

        # Initialize model and move it to the appropriate device
        model_config = {"type": model_type, "task": experimental_setting.model.task, "num_classes": num_classes}
        model = get_3d_model(
            model_config=model_config,
            hyperparameters=hyperparameters,
            developer_mode=experimental_setting.developer_mode,
            spatial_size=spatial_size,
            is_medmnist=dataset.get("is_medmnist", False)
        ).to(device)

        print(f"\nModel initialized: {model_type}")

        # Create test data in the format expected by 3D dataloaders
        test_data = [{"index": idx, "image": img, "label": label} 
                           for idx, (img, label) in enumerate(zip(dataset["test_images"], dataset["test_labels"]))]
        
        # Storage for cross-fold evaluation
        # - We collect per-fold class probabilities for the entire test set
        # - We keep ground-truth targets only once (from the first fold iteration)
        # - We also keep per-fold metric summaries for optional reporting
        folds_probabilities = []
        ground_truth_targets = None  # Ground truth test labels are identical across folds, so we only need to store them once

        # Calculate total inner folds (repeats * splits)
        cv_inner_folds_splits = experimental_setting.cv_inner_folds_splits if hasattr(experimental_setting, "cv_inner_folds_splits") else 5
        cv_inner_folds_repeats = experimental_setting.cv_inner_folds_repeats if hasattr(experimental_setting, "cv_inner_folds_repeats") else 1
        total_inner_folds = cv_inner_folds_repeats * cv_inner_folds_splits
        
        # Evaluate each fold's model on the complete test set
        for fold in range(total_inner_folds):
            # Load normalization stats from the inner fold's normalization_stats.txt file
            normalization_stats = load_normalization_stats_from_fold(pipeline_directory, fold)
            
            # For MRI datasets (lipo, desmoid, liver), normalization_stats is None because
            # normalization is done per image/patient in preprocessing. This is expected.
            dataset_name = experimental_setting.data.dataset.lower()
            is_mri_dataset = dataset_name in ["lipo", "desmoid", "liver"]
            
            if normalization_stats is None:
                if is_mri_dataset:
                    print(f"Note: No normalization stats file found for fold {fold} (expected for MRI datasets)")
                else:
                    print(f"Warning: Normalization stats not found for fold {fold} (CT dataset), skipping...")
                    continue
            else:
                print(f"Normalization stats: {normalization_stats}")

            # Create test dataset with transforms (no augmentation for evaluation)
            test_dataset = Dataset(
                test_data, 
                transform=DataTransform(normalization_stats, developer_mode=experimental_setting.developer_mode, spatial_size=spatial_size, is_training=False, is_medmnist=dataset.get("is_medmnist", False), augmentation_type=experimental_setting.data.augmentation_type)
            )
            
            # Worker init function to ensure each worker has a deterministic seed
            # This is critical when num_workers > 0 for reproducibility
            def worker_init_fn(worker_id):
                # Set seed for each worker based on the base seed and worker ID
                # This ensures reproducibility even with multiple workers
                worker_seed = experimental_setting.seed + worker_id
                set_seed(worker_seed)

            # Create test loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=hyperparameters.get(
                    "batch_size",
                    getattr(experimental_setting.training, "batch_size", 1)
                ),
                shuffle=False,
                num_workers=experimental_setting.data.num_workers,
                pin_memory=False,
                worker_init_fn=worker_init_fn if experimental_setting.data.num_workers > 0 else None,  # Deterministic workers
            )

            # EVALUATE THE FOLD ON THE TEST SET
            # ------------------------------------------------------------------------------------------------
            # Evaluate the fold on the test set
            print(f"\n=== Evaluating Fold {fold + 1}/{total_inner_folds} on Test Set ===")
            fold_probabilities, fold_targets = evaluate_fold(
                fold, test_loader, model, experimental_setting, hyperparameters, pipeline_directory
            )
            
            # Skip if checkpoint not found
            if fold_probabilities is None or fold_targets is None:
                print(f"Skipping fold {fold + 1} due to missing checkpoint")
                continue
            
            # After finishing this fold, store probabilities and ground truth targets
            folds_probabilities.append(np.asarray(fold_probabilities))
            if ground_truth_targets is None:
                ground_truth_targets = np.asarray(fold_targets)

    else:
        raise ValueError(f"Unsupported dimensionality: {experimental_setting.data.dimensionality.lower()}. Must be either '2d' or '3d'")
    
    # CALCULATE METRICS
    # ------------------------------------------------------------------------------------------------
    if not folds_probabilities:
            print("Warning: No valid checkpoints found for evaluation")
            return None
    
    # 1. Calculate Ensemble Metrics
    ensemble_metrics = calculate_ensemble_metrics(folds_probabilities, ground_truth_targets, num_classes)
    
    # 2. Calculate Per-Fold Metrics
    per_fold_metrics = calculate_per_fold_metrics(folds_probabilities, ground_truth_targets, num_classes)

    # 3. Post-hoc: Calculate ensemble probabilities for outer fold ensemble
    stacked_probabilities = np.stack(folds_probabilities, axis=0)  # (folds, samples, classes)
    avg_probabilities = np.mean(stacked_probabilities, axis=0)     # (samples, classes)

    # Save raw predictions for outer fold ensemble
    prediction_data = {
        "probabilities": avg_probabilities.tolist(),
        "ground_truth": ground_truth_targets.tolist(),
        "sample_ids": list(range(len(ground_truth_targets))),
        "cv_outer_fold": cv_outer_fold,
        "num_classes": num_classes,
        "framework": framework
    }
    
    predictions_file = os.path.join(pipeline_directory, "test_predictions_for_outer_ensemble.json")
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=4)
    print(f"Raw predictions saved for outer fold ensemble: {predictions_file}")
    

    # Return both ensemble metrics and per-fold metrics
    return {
        "ensemble": ensemble_metrics,
        "per_fold": per_fold_metrics,
    }  
    
    
