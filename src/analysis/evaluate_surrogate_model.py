"""
Evaluate Surrogate Model (MLP) performance on portfolio data using Cross-Validation.

This script evaluates how well QuickTune's surrogate model can predict performance
and cost from portfolio configurations. It performs K-Fold Cross-Validation to
assess generalization capability.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.quicktune_utils import (
    PortfolioManager,
    CustomPerfPredictor,
    CustomCostPredictor,
    FTPFNPerfPredictor,
)
from src.utils.portfolio_preprocessing import preprocess_portfolio_for_quicktune


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with MAE, RMSE, and R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
    }


def evaluate_predictor_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    predictor_class,
    predictor_kwargs: Dict,
    pipeline_space_path: str,
    n_folds: int = 5,
    random_state: int = 42,
    is_performance_predictor: bool = False,
    groups: np.ndarray = None
) -> Dict[str, List[float]]:
    """
    Evaluate predictor using K-Fold Cross-Validation.
    
    Args:
        X: Feature matrix (portfolio configurations)
        y: Target values (performance curves or costs)
        predictor_class: Predictor class to use (CustomPerfPredictor, CustomCostPredictor, etc.)
        predictor_kwargs: Additional keyword arguments for predictor initialization
        pipeline_space_path: Path to pipeline space YAML file
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility
        is_performance_predictor: Whether this is a performance predictor (needs curve parameter)
        groups: Group labels for GroupKFold (e.g., dataset names). If provided, ensures
                all samples from the same group are in the same fold (prevents data leakage).
        
    Returns:
        Dictionary with metrics for each fold and mean/std across folds
    """
    # Set random seeds for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    try:
        import torch
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    except ImportError:
        pass
    
    # Use GroupKFold if groups are provided (e.g., to keep datasets together)
    # Otherwise use regular KFold
    # NOTE: GroupKFold doesn't have random_state, but it's deterministic based on data order
    # We keep the original data order (as loaded from CSV) to maintain consistency
    if groups is not None:
        kf = GroupKFold(n_splits=n_folds)
        print(f"Using GroupKFold: {len(np.unique(groups))} unique groups")
        print(f"Random seed set to {random_state} for reproducibility")
        print(f"Note: GroupKFold splits are deterministic based on data order (as loaded from CSV)")
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        print("Using KFold with shuffling")
    
    fold_metrics = {
        "MAE": [],
        "RMSE": [],
        "R²": [],
    }
    
    # For performance curves, we need to predict the entire curve
    # We'll use the mean of the curve as the target for evaluation
    if len(y.shape) > 1 and y.shape[1] > 1:
        # y is a 2D array (curves), use mean performance
        y_target = np.mean(y, axis=1)
        print(f"Using mean performance across {y.shape[1]} epochs as target")
    else:
        # y is 1D (costs or single values)
        y_target = y.flatten()
    
    # Determine split iterator based on whether groups are provided
    if groups is not None:
        # Use original data order (GroupKFold is deterministic if data order is consistent)
        split_iterator = kf.split(X, groups=groups)
    else:
        split_iterator = kf.split(X)
    
    for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Split data
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y_target[val_idx]
        
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        
        # Show detailed dataset distribution in this fold (if groups are available)
        if groups is not None:
            train_datasets = pd.Series(groups[train_idx]).value_counts().sort_index()
            val_datasets = pd.Series(groups[val_idx]).value_counts().sort_index()
            
            print(f"\nTrain datasets ({len(train_datasets)} datasets):")
            for dataset, count in train_datasets.items():
                print(f"  {dataset}: {count} samples")
            print(f"\nVal datasets ({len(val_datasets)} datasets):")
            for dataset, count in val_datasets.items():
                print(f"  {dataset}: {count} samples")
            
            # Show which datasets are in which set
            print(f"\nDataset split summary:")
            all_datasets = sorted(set(groups))
            for dataset in all_datasets:
                train_count = (groups[train_idx] == dataset).sum()
                val_count = (groups[val_idx] == dataset).sum()
                total = train_count + val_count
                train_pct = (train_count / total * 100) if total > 0 else 0
                val_pct = (val_count / total * 100) if total > 0 else 0
                print(f"  {dataset}: {train_count} train ({train_pct:.1f}%), {val_count} val ({val_pct:.1f}%)")
        
        # Preprocess training data
        X_train_processed = preprocess_portfolio_for_quicktune(
            df=X_train,
            pipeline_space_path=pipeline_space_path,
            add_active_flags=True,
            handle_inactive_categorical=True,
            inactive_categorical_value="__inactive__"
        )
        
        # Train predictor
        predictor = predictor_class(**predictor_kwargs)
        
        try:
            predictor.fit(X=X_train_processed, y=y_train)
            print("Predictor trained successfully")
        except Exception as e:
            print(f"Error training predictor: {e}")
            continue
        
        # Preprocess validation data
        X_val_processed = preprocess_portfolio_for_quicktune(
            df=X_val,
            pipeline_space_path=pipeline_space_path,
            add_active_flags=True,
            handle_inactive_categorical=True,
            inactive_categorical_value="__inactive__"
        )
        
        # Predict
        try:
            # PerfPredictor needs curve parameter, CostPredictor doesn't
            if is_performance_predictor:
                # For performance predictor, we need to provide a dummy curve
                # Since we're evaluating on portfolio data, we can use the validation curves
                # or create dummy curves with the expected shape
                if len(y.shape) > 1 and y.shape[1] > 1:
                    # Use validation curves
                    val_curves = y[val_idx]
                    pred_result = predictor.predict(X=X_val_processed, curve=val_curves)
                else:
                    # Single values - create dummy curve (shouldn't happen for perf predictor)
                    dummy_curve = np.ones((len(X_val), 1)) * 0.5  # Dummy curve
                    pred_result = predictor.predict(X=X_val_processed, curve=dummy_curve)
            else:
                # CostPredictor: no curve needed
                pred_result = predictor.predict(X=X_val_processed)
            
            if isinstance(pred_result, tuple):
                # PerfPredictor: returns (mean, std)
                y_pred, _ = pred_result
                y_pred = np.array(y_pred).flatten()
            else:
                # CostPredictor: returns array directly
                y_pred = np.array(pred_result).flatten()
            
            print(f"Predictions shape: {y_pred.shape}, True values shape: {y_val.shape}")
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, y_pred)
            
            print(f"Fold {fold_idx + 1} Metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
                fold_metrics[metric_name].append(metric_value)
                
        except Exception as e:
            print(f"Error predicting: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate mean and std across folds
    results = {}
    for metric_name in fold_metrics.keys():
        if fold_metrics[metric_name]:
            results[f"{metric_name}_mean"] = np.mean(fold_metrics[metric_name])
            results[f"{metric_name}_std"] = np.std(fold_metrics[metric_name])
            results[f"{metric_name}_folds"] = fold_metrics[metric_name]
        else:
            results[f"{metric_name}_mean"] = np.nan
            results[f"{metric_name}_std"] = np.nan
            results[f"{metric_name}_folds"] = []
    
    return results


def evaluate_portfolio(
    portfolio_dir: str,
    pipeline_space_path: str,
    n_folds: int = 5,
    model_type: str = "gp",
    seed: int = 42,
    output_dir: str = None
) -> Dict:
    """
    Evaluate surrogate model performance on a portfolio.
    
    Args:
        portfolio_dir: Path to portfolio directory
        pipeline_space_path: Path to pipeline space YAML file
        n_folds: Number of CV folds
        model_type: Type of surrogate model to use. Options:
            - "gp": Gaussian Process (default, slower but accurate)
            - "ftpfn": FT-PFN (faster, neural network based)
        seed: Random seed
        output_dir: Optional output directory for results
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Portfolio: {portfolio_dir}")
    print(f"{'='*80}\n")
    
    # Load portfolio
    portfolio = PortfolioManager.load(portfolio_dir)
    
    print(f"Portfolio loaded:")
    print(f"  Configurations: {len(portfolio.pipeline_df)}")
    print(f"  Curves shape: {portfolio.curve_df.shape}")
    print(f"  Costs shape: {portfolio.cost_df.shape}")
    print(f"  Meta features: {len(portfolio.meta_df)} datasets")
    
    # Prepare data
    # Merge pipeline configs with metadata (similar to run_quicktune.py)
    merged_df = pd.merge(
        portfolio.pipeline_df,
        portfolio.meta_df.drop_duplicates(subset=['dataset']),
        how='left',
    )
    
    # Extract dataset groups for GroupKFold (before removing dataset column)
    # This ensures all configurations from the same dataset stay in the same fold
    groups = None
    if "dataset" in merged_df.columns:
        groups = merged_df["dataset"].values
        print(f"Found dataset column: {len(np.unique(groups))} unique datasets")
        print(f"Dataset distribution: {pd.Series(groups).value_counts().to_dict()}")
        # Remove dataset identifier from features (but keep groups for splitting)
        merged_df = merged_df.drop(columns=["dataset"])
    else:
        print("Warning: No 'dataset' column found. Using regular KFold (may cause data leakage if data is sorted by dataset).")
    
    # Remove number_of_epochs if present (not a hyperparameter)
    if "number_of_epochs" in merged_df.columns:
        merged_df = merged_df.drop(columns=["number_of_epochs"])
    
    # Convert to numpy arrays
    curve = portfolio.curve_df.values
    cost = portfolio.cost_df.values
    
    print(f"\nData prepared:")
    print(f"  Features: {merged_df.shape[1]} columns")
    print(f"  Samples: {len(merged_df)}")
    print(f"  Curve shape: {curve.shape}")
    print(f"  Cost shape: {cost.shape}")
    
    # Prepare predictor arguments
    predictor_kwargs = {
        "path": None,  # Don't save during CV
        "seed": seed,
        "pipeline_space_path": pipeline_space_path
    }
    
    # Evaluate performance predictor
    print(f"\n{'='*80}")
    print("Evaluating Performance Predictor")
    print(f"{'='*80}\n")
    
    # Select predictor class based on model_type
    model_type = model_type.lower()
    if model_type == "ftpfn":
        perf_predictor_class = FTPFNPerfPredictor
        print("Using FT-PFN Performance Predictor (Neural Network, faster)")
    elif model_type == "gp":
        perf_predictor_class = CustomPerfPredictor
        print("Using GP Performance Predictor (Gaussian Process, slower but accurate)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'gp' or 'ftpfn'.")
    
    perf_results = evaluate_predictor_cv(
        X=merged_df,
        y=curve,
        predictor_class=perf_predictor_class,
        predictor_kwargs=predictor_kwargs,
        pipeline_space_path=pipeline_space_path,
        n_folds=n_folds,
        random_state=seed,
        is_performance_predictor=True,
        groups=groups
    )
    
    # Evaluate cost predictor
    print(f"\n{'='*80}")
    print("Evaluating Cost Predictor")
    print(f"{'='*80}\n")
    
    cost_predictor_class = CustomCostPredictor
    print("Using Custom MLP Cost Predictor")
    
    cost_results = evaluate_predictor_cv(
        X=merged_df,
        y=cost,
        predictor_class=cost_predictor_class,
        predictor_kwargs=predictor_kwargs,
        pipeline_space_path=pipeline_space_path,
        n_folds=n_folds,
        random_state=seed,
        is_performance_predictor=False,
        groups=groups
    )
    
    # Compile results
    results = {
        "portfolio_dir": portfolio_dir,
        "n_samples": len(merged_df),
        "n_folds": n_folds,
        "model_type": model_type,
        "performance_predictor": {
            "type": model_type.upper(),
            **perf_results
        },
        "cost_predictor": {
            "type": "MLP",
            **cost_results
        }
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print("Performance Predictor:")
    print(f"  MAE: {perf_results['MAE_mean']:.4f} ± {perf_results['MAE_std']:.4f}")
    print(f"  RMSE: {perf_results['RMSE_mean']:.4f} ± {perf_results['RMSE_std']:.4f}")
    print(f"  R²: {perf_results['R²_mean']:.4f} ± {perf_results['R²_std']:.4f}")
    
    print("\nCost Predictor:")
    print(f"  MAE: {cost_results['MAE_mean']:.4f} ± {cost_results['MAE_std']:.4f}")
    print(f"  RMSE: {cost_results['RMSE_mean']:.4f} ± {cost_results['RMSE_std']:.4f}")
    print(f"  R²: {cost_results['R²_mean']:.4f} ± {cost_results['R²_std']:.4f}")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        portfolio_name = Path(portfolio_dir).name
        results_file = output_path / f"surrogate_evaluation_{portfolio_name}.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QuickTune surrogate model performance on portfolio data using Cross-Validation"
    )
    parser.add_argument(
        "portfolio_dir",
        type=str,
        help="Path to portfolio directory (contains config.csv, curve.csv, cost.csv, meta.csv)"
    )
    parser.add_argument(
        "--pipeline-space",
        type=str,
        required=True,
        help="Path to pipeline space YAML file"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gp",
        choices=["gp", "ftpfn"],
        help="Type of surrogate model to use. 'gp' (Gaussian Process, slower but accurate) or 'ftpfn' (FT-PFN, faster neural network). Default: gp"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results JSON file (default: same as portfolio_dir)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.portfolio_dir).parent)
    
    # Evaluate portfolio
    results = evaluate_portfolio(
        portfolio_dir=args.portfolio_dir,
        pipeline_space_path=args.pipeline_space,
        n_folds=args.n_folds,
        model_type=args.model_type,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()


