import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.classification_2d.models_2d import get_2d_model
from src.classification_2d.preprocess_data_2d import BrainTumorDataset, get_max_batch_size, load_brain_tumor_dataset
from src.classification_3d.models_3d import get_3d_model
from src.classification_3d.preprocess_data_3d import calculate_voxel_size_from_images
from src.classification_3d.preprocess_data_3d import EvaluationTransform
from src.utils.common_utils import set_seed, yaml_to_neps_pipeline_space
from src.utils.model_lifecycle_utils import evaluate_model
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from monai.data import Dataset

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


def evaluate_fold(fold, test_loader, experimental_setting, hyperparameters, num_classes, pipeline_directory, framework="neps"):
    """
    Evaluate a single fold's trained model on the test set and return prediction probabilities.
    
    This function loads a trained model checkpoint for a specific fold and evaluates it on the test set.
    It returns softmax probabilities (not hard predictions) to enable ensemble averaging.
    
    Args:
        fold (int): Fold index (0-based) for which to evaluate the model
        test_loader (DataLoader): PyTorch DataLoader containing test data
        experimental_setting (DictConfig): Hydra configuration object with experiment settings
        hyperparameters (dict): Hyperparameters used for training the model
        num_classes (int): Number of classes in the classification task
        pipeline_directory (str): Directory containing the trained model checkpoints
        framework (str): Framework being used ("neps" or "quicktune"), affects model loading
    
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
        - Model is loaded with fold-specific hyperparameters
        - Works for both NePS and QuickTune frameworks
    """
    # Set device for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MODEL INITIALIZATION
    # ------------------------------------------------------------------------------------------------
    # Initialize the model based on framework
    if experimental_setting.data.dimensionality.lower() == "3d":
        # Use smaller model for baseline run in developer mode
        if experimental_setting.developer_mode and experimental_setting.run_mode == "Baseline":
            hyperparameters["conv0_stride"] = 1
            hyperparameters["init_features"] = 8
            hyperparameters["bn_size"] = 1
            hyperparameters["growth_rate"] = 6
            hyperparameters["num_layers_block1"] = 2
            hyperparameters["num_layers_block2"] = 4
            hyperparameters["num_layers_block3"] = 8
            hyperparameters["num_layers_block4"] = 4
        if framework == "quicktune" and "model" in hyperparameters:
            # QuickTune: model type is in hyperparameters
            model = get_3d_model(
                {
                    "type": hyperparameters["model"],
                    "task": experimental_setting.model.task,
                    "num_classes": num_classes,
                }, 
                hyperparameters
            )
        else:
            # NePS: model type is in experimental_setting
            model = get_3d_model(
                {
                    "type": experimental_setting.model.type,
                    "task": experimental_setting.model.task,
                    "num_classes": num_classes,
                }, 
                hyperparameters
            )
    else:
        raise NotImplementedError("2D evaluation is not supported for config evaluation yet.")
    
    # MODEL CHECKPOINT LOADING
    # ------------------------------------------------------------------------------------------------
    # Load the trained model checkpoint for this fold
    checkpoint_path = (
        Path(pipeline_directory)
        / f"cv_inner_fold_{fold}"
        / "best_model_checkpoint.pth"
    )
    
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        return None, None
    
    # Add safe globals for NumPy objects that might be in the checkpoint
    # TODO @Diane: check this out!
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
    
    # Set seed for reproducibility
    set_seed(experimental_setting.seed)

    # TEST DATA LOADING
    # ------------------------------------------------------------------------------------------------
    # Load test data based on dimensionality
    if experimental_setting.data.dimensionality.lower() == "2d":
        raise NotImplementedError("2D evaluation is not supported for config evaluation yet.")
        
    elif experimental_setting.data.dimensionality.lower() == "3d":
        # Determine voxel calculation method
        voxel_calc = hyperparameters.get("voxel_calculation", "median")
        # TODO @Diane: delete: dataset_dict_key = f"dataset_dict_{voxel_calc}"
        dataset_dict = dataset_dict if experimental_setting.run_mode == "Baseline" else dataset_dict[f"dataset_dict_{voxel_calc}"]
        
        # Create test data in the format expected by 3D dataloaders
        test_data = [{"index": idx, "image": img, "label": label} 
                           for idx, (img, label) in enumerate(zip(dataset_dict["test_images"], dataset_dict["test_labels"]))]
        
        # Get voxel size for the dataset
        voxel_size = dataset_dict["voxel_size"]

        # Storage for cross-fold evaluation
        # - We collect per-fold class probabilities for the entire test set
        # - We keep ground-truth targets only once (from the first fold iteration)
        # - We also keep per-fold metric summaries for optional reporting
        folds_probabilities = []
        ground_truth_targets = None  # Ground truth test labels are identical across folds, so we only need to store them once

        # Evaluate each fold's model on the complete test set
        for fold in range(experimental_setting.cv_inner_folds):
            # Normalization stats are calculated from the preprocessed training data separately for each inner fold!
            normalization_stats = None  # TODO @Diane: get normalization stats based on the inner fold!

            # Create test dataset with transforms (no augmentation for evaluation)
            test_dataset = Dataset(
                test_data, 
                transform=EvaluationTransform(voxel_size, normalization_stats, developer_mode=experimental_setting.developer_mode)
            )
            
            # Create test loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=hyperparameters.get("batch_size", 1),
                shuffle=False,
                num_workers=experimental_setting.data.num_workers,
            )

            # EVALUATE THE FOLD ON THE TEST SET
            # ------------------------------------------------------------------------------------------------
            # Evaluate the fold on the test set
            print(f"\n=== Evaluating Fold {fold + 1}/{experimental_setting.cv_inner_folds} on Test Set ===")
            fold_probabilities, fold_targets = evaluate_fold(
                fold, test_loader, experimental_setting, hyperparameters, num_classes, pipeline_directory, framework
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
        raise ValueError(f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'")
    
    # CALCULATE METRICS
    # ------------------------------------------------------------------------------------------------
    if not folds_probabilities:
            print("Warning: No valid checkpoints found for evaluation")
            return None
    
    # 1. Calculate Ensemble Metrics
    ensemble_metrics = calculate_ensemble_metrics(folds_probabilities, ground_truth_targets, num_classes)
    
    # 2. Calculate Per-Fold Metrics
    per_fold_metrics = calculate_per_fold_metrics(folds_probabilities, ground_truth_targets, num_classes)

    # Return both ensemble metrics and per-fold metrics
    return {
        "ensemble": ensemble_metrics,
        "per_fold": per_fold_metrics,
    }  
    
    
