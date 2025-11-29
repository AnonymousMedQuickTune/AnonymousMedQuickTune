import logging
import os
import pickle
from pathlib import Path
import warnings
import tempfile
import csv

import hydra
import json
import yaml
import numpy as np
from neps import run
from omegaconf import DictConfig, OmegaConf

# Suppress multiprocessing cleanup warnings
warnings.filterwarnings("ignore", message=".*Directory not empty.*")

from src.classification_2d.objective_function_2d import run_2d_pipeline
from src.classification_2d.preprocess_data_2d import load_brain_tumor_dataset
from src.classification_3d.objective_function_3d import run_3d_pipeline
from src.classification_3d.preprocess_data_3d import load_3d_dataset_with_outer_cv_splits
from src.utils.common_utils import (get_cache_file_path, neps_space_to_dict, set_seed,
                                    yaml_to_neps_pipeline_space, cleanup_training_artifacts)
from src.utils.experiment_status_logger import ExperimentStatusLogger
from src.utils.logging_utils import (save_cv_summary, update_performances_csv_from_neps_output,
                                     update_cost_csv_from_neps_output)
from src.evaluate_trained_config import evaluate_config_on_test_set
from src.analysis.summarize_evaluation_results import summarize_experiment


def run_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    experimental_setting,
    dataset_dict,
    num_classes,
    **hyperparameters,
):
    """
    Main pipeline function that delegates to specific 2D or 3D implementations
    based on experimental_setting.data.dimensionality and evaluates the trained
    configuration on the test set.

    NOTE: The argument order and parameter names must strictly follow NePS conventions
    to ensure proper optimization and checkpointing functionality.

    Args:
        pipeline_directory (str): Directory where current pipeline results will be saved
        previous_pipeline_directory (str): Directory containing previous pipeline runs
        experimental_setting (DictConfig): Hydra configuration object
        dataset_dict (dict, optional): Combined train+val data and labels dictionary if preloaded
        num_classes (int, optional): Number of classes in the dataset if preloaded
        **hyperparameters: Configuration dictionary containing hyperparameters

    Returns:
        dict: Dictionary containing optimization metrics for NePS
        
    Process:
        1. Run the appropriate training pipeline (2D or 3D) with cross-validation
        2. Evaluate the trained configuration on the test set using ensemble predictions
        3. Save test evaluation results to test_evaluation_results.json
        4. Clean up training artifacts (model checkpoints) to save disk space
        5. Return pipeline result to NePS for optimization
    """
    # Extract CV fold from pipeline directory path to use fold-specific seed
    pipeline_dir_str = str(pipeline_directory)
    cv_outer_fold = 0  # Default to 0 if not found in path
    if "cv_outer_fold_" in pipeline_dir_str:
        try:
            cv_outer_fold = int(pipeline_dir_str.split("cv_outer_fold_")[-1].split("/")[0])
        except (ValueError, IndexError):
            cv_outer_fold = 0
    
    # Set fold-specific seed for pipeline reproducibility
    fold_specific_seed = experimental_setting.seed + cv_outer_fold
    set_seed(fold_specific_seed)
    
    # Extract config number from pipeline directory to print in the console
    config_number = pipeline_dir_str.split('/configs/config_')[-1] if '/configs/config_' in pipeline_dir_str else "unknown"
    print(f"\n{'-' * 100}")
    if "baseline" in str(experimental_setting.pipeline_space):
        print(f"Running Baseline configuration")
    else:
        print(f"Running NePS configuration #{config_number}/{experimental_setting.max_evaluations}")
    print(f"{'-' * 100}\n")

    dimensionality = experimental_setting.data.dimensionality.lower()

    # Run the appropriate training pipeline (2D or 3D)
    if dimensionality == "2d":
        pipeline_result = run_2d_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            experimental_setting=experimental_setting,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            **hyperparameters,
        )
    elif dimensionality == "3d":
        pipeline_result = run_3d_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            experimental_setting=experimental_setting,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            **hyperparameters,
        )
    else:
        raise ValueError(
            f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
        )
    
    # Save pipeline_result to a file
    with open(os.path.join(pipeline_directory, "pipeline_result.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline_result, f, indent=4)
    
    # Evaluate the trained configuration on test set
    print(f"\n{'='*100}")
    print(f"STARTING TEST SET EVALUATION FOR CURRENT CONFIG")
    print(f"{'='*100}\n")
    
    
    # Evaluate configuration on test set
    test_metrics = evaluate_config_on_test_set(
        pipeline_directory=pipeline_directory,
        experimental_setting=experimental_setting,
        dataset_dict=dataset_dict,
        num_classes=num_classes,
        hyperparameters=hyperparameters,
        cv_outer_fold=cv_outer_fold
    )

    # Persist test metrics and validation metrics as a JSON artifact; do not modify pipeline_result or report.yaml
    if test_metrics is not None:
        # Extract validation metrics from pipeline_result
        validation_metrics = None
        if "extra" in pipeline_result and "all_folds_final_metrics" in pipeline_result["extra"]:
            validation_metrics = pipeline_result["extra"]["all_folds_final_metrics"]
        
        # Combine test and validation metrics
        # Keep backward compatibility: test_metrics already has "ensemble" and "per_fold" structure
        evaluation_results = test_metrics.copy()  # Start with test metrics structure
        
        # Add validation metrics if available
        if validation_metrics is not None:
            evaluation_results["validation"] = validation_metrics
            print(f"Validation metrics included: {validation_metrics}")
        
        test_metrics_file = os.path.join(pipeline_directory, "test_evaluation_results.json")
        with open(test_metrics_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Test and validation evaluation completed and saved to: {test_metrics_file}")
    else:
        print(f"Warning: Test evaluation failed or no valid checkpoints found!")
    
    # Update incumbent_performances.csv after test evaluation completes
    # Extract NePS output directory from pipeline directory
    pipeline_dir_path = Path(pipeline_directory)
    if "NePS_output" in str(pipeline_dir_path):
        # Find NePS_output directory by going up the path
        current_path = pipeline_dir_path
        while current_path.name != "NePS_output" and current_path.parent != current_path:
            current_path = current_path.parent
        if current_path.name == "NePS_output":
            # Get outer fold directory (parent of configs)
            outer_fold_dir = pipeline_dir_path.parent.parent  # .../cv_outer_fold_X
            update_performances_csv_from_neps_output(str(outer_fold_dir), cv_outer_fold)
    
    # Delete model checkpoints to save disk space  # TODO @Diane: Keep incumbent model checkpoint?
    # NOTE: After test evaluation, model checkpoints are no longer needed
    # Calculate total inner folds (repeats * splits)
    cv_inner_folds_splits = experimental_setting.cv_inner_folds_splits if hasattr(experimental_setting, "cv_inner_folds_splits") else 5
    cv_inner_folds_repeats = experimental_setting.cv_inner_folds_repeats if hasattr(experimental_setting, "cv_inner_folds_repeats") else 1
    total_inner_folds = cv_inner_folds_repeats * cv_inner_folds_splits
    cleanup_training_artifacts(pipeline_directory, total_inner_folds)
    
    # Print pipeline result and test metrics
    print(f"\n\nPipeline result: {pipeline_result}")
    print(f"\nTest metrics: {test_metrics}\n\n")
    
    # Automatically generate performance plots
    try:
        # Extract experiment directory from pipeline_directory
        # Both NePS and Baseline experiments have the same structure with /NePS_output/
        # Examples:
        # - NePS: experiments/NePS/lipo/test_plotting_script/seed_42/NePS_output/cv_outer_fold_0/configs/config_3/...
        # - Baseline: experiments/Baseline/liver/test_liver_31/seed_42/NePS_output/cv_outer_fold_0/configs/config_1/...
        # Experiment directory: experiments/NePS/lipo/test_plotting_script or experiments/Baseline/liver/test_liver_31
        pipeline_dir_str = str(pipeline_directory)
        if "/NePS_output/" in pipeline_dir_str:
            # Extract path up to NePS_output, then go up one level to get experiment directory
            experiment_dir_str = pipeline_dir_str.split("/NePS_output/")[0]
            experiment_dir = Path(experiment_dir_str).parent  # Go up from seed_42 to experiment directory
            
            # Check if experiment directory exists and has the expected structure
            if experiment_dir.exists() and any(experiment_dir.iterdir()):
                from src.analysis.plot_results_over_time import collect_performances, create_plots
                
                print(f"\n{'='*100}")
                print(f"GENERATING PERFORMANCE PLOTS")
                print(f"{'='*100}\n")
                
                # Collect performances and create plots (single experiment)
                validation_performances, test_performances = collect_performances(experiment_dir)
                all_validation_performances = [(experiment_dir.name, validation_performances)]
                all_test_performances = [(experiment_dir.name, test_performances)]
                create_plots([experiment_dir], all_validation_performances, all_test_performances)
                
                print(f"Performance plots generated successfully!\n")
    except Exception as e:
        # Don't fail the pipeline if plotting fails
        print(f"Warning: Could not generate performance plots: {e}")
        print("Continuing with pipeline execution...\n")
    
    # Return the pipeline result to NePS
    return pipeline_result

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experimental_setting.yaml",
)
def main(experimental_setting: DictConfig) -> None:
    """
    Main entry point for the NePStraining script.

    Args:
        experimental_setting (DictConfig): Hydra configuration object
    """
    # Set seed for NePS reproducibility
    set_seed(experimental_setting.seed)

    if experimental_setting.run_mode == "Baseline":
        print(f"\n\nBaseline run!\n\n")
        experimental_setting.max_evaluations = 1
        experimental_setting.pipeline_space = "configs/pipeline_spaces/baseline.yaml"
        experimental_setting.data.voxel_calculation = "median"

    if experimental_setting.developer_mode:
        print(f"\n\nDeveloper mode is enabled!\n\n")
        if experimental_setting.run_mode != "Baseline":
            experimental_setting.max_evaluations = 4
            experimental_setting.pipeline_space = "configs/pipeline_spaces/efficientnet.yaml"
        experimental_setting.training.number_of_epochs = 2
        # Set number of inner CV folds for developer mode: #repeats * #splits per repeat = #total inner folds
        experimental_setting.cv_inner_folds_repeats = 1
        experimental_setting.cv_inner_folds_splits = 2
        # Set number of outer CV folds for developer mode: #repeats * #splits per repeat = #total outer folds
        experimental_setting.cv_outer_folds_repeats = 1
        experimental_setting.cv_outer_folds_splits = 2  # splits  (minimum!) per repeat
    
    # Calculate total inner folds (repeats * splits)
    cv_inner_folds_splits = experimental_setting.cv_inner_folds_splits if hasattr(experimental_setting, "cv_inner_folds_splits") else 5
    cv_inner_folds_repeats = experimental_setting.cv_inner_folds_repeats if hasattr(experimental_setting, "cv_inner_folds_repeats") else 1
    total_inner_folds = cv_inner_folds_repeats * cv_inner_folds_splits

    # If no validation set is used, set total inner folds to 1
    if experimental_setting.training.no_validation:
        total_inner_folds = 1
        if experimental_setting.run_mode != "Baseline":
            raise ValueError("No validation set mode is not supported for non-baseline runs.")

    # TODO @Diane: Double check training search space!
    # TODO @Diane: Implement SwinUNETR search space!
    # TODO @Diane: Test all search spaces for the different datasets!
    # TODO @Diane: Implement Baseline integration to Portfolio!
    # Combine model and training space into a single search space
    if experimental_setting.combine_model_and_training_space and not any(x in experimental_setting.pipeline_space for x in ["baseline", "training"]):
        print(f"\n\nCombining model and training spaces!\n\n")
        
        # Load training space configuration
        training_space_path = "configs/pipeline_spaces/training.yaml"
        try:
            with open(training_space_path, "r", encoding="utf-8") as f:
                training_space = yaml.safe_load(f)
        except (yaml.YAMLError, IOError) as e:
            logging.error(f"Failed to load training space configuration: {e}")
            raise
        
        # Load model space configuration
        try:
            with open(experimental_setting.pipeline_space, "r", encoding="utf-8") as f:
                model_space = yaml.safe_load(f)
        except (yaml.YAMLError, IOError) as e:
            logging.error(f"Failed to load model space / AutoNorm space configuration: {e}")
            raise
        
        # Combine spaces (model space takes precedence for overlapping keys)
        combined_space = {**training_space, **model_space}
        
        # Save combined space to a temporary file
        # Extract original config name for better filename
        original_config_name = os.path.basename(experimental_setting.pipeline_space).replace('.yaml', '')
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_training_combined_{original_config_name}.yaml', delete=False) as tmp_file:
            yaml.dump(combined_space, tmp_file, default_flow_style=False)
            experimental_setting.pipeline_space = tmp_file.name
        
        print(f"\nCombined space: {experimental_setting.pipeline_space}")
        print(f"Combined space contains {len(combined_space)} parameters:")
        print(f"- Training parameters: {list(training_space.keys())}")
        print(f"- Model/AutoNorm parameters: {list(model_space.keys())}")
        print(f"- Combined parameters: {list(combined_space.keys())}\n")

    # Convert YAML pipeline space configuration into NePS-compatible format
    # NePS requires a specific dictionary structure for hyperparameter definitions
    pipeline_space = yaml_to_neps_pipeline_space(experimental_setting.pipeline_space)

    # Print experimental setting and pipeline space
    print("\nexperimental setting: ", experimental_setting, "\n\npipeline space: ", pipeline_space, "\n")
    
    # Validate that multifidelity searchers have a fidelity parameter in the pipeline space
    if experimental_setting.searcher in ["priorband", "hyperband", "asha", "async_hb", "successive_halving", "ifbo"]:
        # Check if any parameter in the pipeline space has is_fidelity=True
        has_fidelity = any(
            hasattr(param, 'is_fidelity') and param.is_fidelity 
            for param in pipeline_space.values()
        )
        if not has_fidelity:
            raise ValueError(
                f"Multifidelity searcher '{experimental_setting.searcher}' requires a fidelity parameter in the pipeline space. "
                f"Please ensure your pipeline space YAML file contains a parameter with 'is_fidelity: true'. "
                f"Current pipeline space: {experimental_setting.pipeline_space}"
            )

    # Create directory for configuration files and logs
    output_dir = os.path.join(experimental_setting.experiment_base_dir, "hydra_output")
    os.makedirs(output_dir, exist_ok=True)

    # Load original pipeline space configuration for human-readable logging
    # This version maintains the original YAML structure without NePS-specific transformations
    with open(experimental_setting.pipeline_space, "r", encoding="utf-8") as f:
        original_pipeline_space = yaml.safe_load(f)

    # Save different versions of configurations:
    # 1. Full Hydra config (includes all settings)
    # 2. NePS-compatible pipeline space (used for optimization)
    # 3. Original compact pipeline space (for better readability)
    # Convert pipeline space to dictionary for YAML serialization
    pipeline_space_dict = neps_space_to_dict(pipeline_space)
    config_files = [
        ("experimental_setting.yaml", OmegaConf.to_yaml(experimental_setting)),
        (
            "pipeline_space.yaml",
            yaml.dump(pipeline_space_dict, default_flow_style=False),
        ),
        (
            "pipeline_space_compact.yaml",
            yaml.dump(original_pipeline_space, default_flow_style=False),
        ),
    ]

    for filename, data in config_files:
        config_path = os.path.join(output_dir, filename)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(data)
        except IOError as e:
            logging.error(f"Failed to write configuration file {filename}: {e}")

    # Handle data loading based on preload configuration
    dataset_dict = None
    num_classes = None

    # Calculate total outer CV folds for N-repeated 3-fold stratified cross-validation
    cv_outer_folds = experimental_setting.cv_outer_folds_repeats * experimental_setting.cv_outer_folds_splits
    
    print(f"\n=== Using N-repeated 3-fold stratified cross-validation ===")
    print(f"N repeats: {experimental_setting.cv_outer_folds_repeats}")
    print(f"N splits per repeat: {experimental_setting.cv_outer_folds_splits}")
    print(f"Total CV folds: {cv_outer_folds}")
    
    # Initialize experiment status logger for webapp dashboard
    status_logger = ExperimentStatusLogger(experimental_setting.experiment_base_dir, experiment_type="neps")
    
    # Set the total number of outer folds for cross-validation to calculate overall progress percentages
    status_logger.set_total_outer_folds(cv_outer_folds)
    
    # Force save the initial status immediately after initialization to ensure the webapp can display "Active" status
    # The initial status file contains:
    # - started timestamp
    # - total_outer_folds count
    # - all outer folds marked as "not_started"
    # - empty outer_folds_progress dictionary
    status_logger._save_main_status()
    
    print(f"\n=== Starting Cross-Validation with {cv_outer_folds} outer folds ===\n")
    
    for cv_outer_fold in range(cv_outer_folds):        
        # Set a different seed for each outer fold to ensure different NePS configurations are sampled
        # This prevents identical hyperparameter configurations across different outer folds
        fold_specific_seed = experimental_setting.seed + cv_outer_fold
        set_seed(fold_specific_seed)
        print(f"\n=== Setting fold-specific seed {fold_specific_seed} for outer fold {cv_outer_fold} ===")
        
        # Load data for current CV fold
        if experimental_setting.data.use_smart_preprocessing:
            print(f"\n{'=' * 100}")
            print(f"Preloading data for outer cross-validation fold {cv_outer_fold + 1}/{cv_outer_folds} based on the selected voxel calculation method")
            print(f"{'=' * 100}\n")
            # Reload data with current CV fold
            dimensionality = experimental_setting.data.dimensionality.lower()
            if dimensionality == "2d":
                if experimental_setting.data.cache_data:
                    raise NotImplementedError("Cross-validation for 2D datasets is not implemented yet for cache_data=True.")
                else:
                    if cv_outer_folds > 1:
                        raise NotImplementedError("Cross-validation for 2D datasets is not implemented yet.")
                    else:
                        if experimental_setting.data.dataset == "brain_tumor":
                            dataset_dict = load_brain_tumor_dataset(
                                data_path=experimental_setting.data.path, seed=experimental_setting.seed, cv_outer_fold=cv_outer_fold
                            )
                        else:
                            raise ValueError(f"Unsupported dataset: {experimental_setting.data.dataset}.")
            elif dimensionality == "3d":
                if experimental_setting.data.cache_data:
                    # Try to load from cache first
                    data_path = Path(experimental_setting.data.path)
                    cache_file = get_cache_file_path(
                        data_path, 
                        experimental_setting.data.dataset, 
                        "3d", 
                        cv_outer_fold, 
                        experimental_setting.data.voxel_calculation
                    )

                    # Cache mechanism further improves performance by avoiding repeated data processing
                    if cache_file.exists():
                        print(f"> Loading 3D data from cache for outer cross-validation fold {cv_outer_fold + 1}/{cv_outer_folds}...")
                        with open(cache_file, "rb") as f:
                            cached_data = pickle.load(f)
                            dataset_dict = cached_data["dataset_dict"]
                            num_classes = cached_data["num_classes"]
                    else:
                        print(f"> No cache found for outer cross-validation fold {cv_outer_fold + 1}/{cv_outer_folds}. Loading 3D data directly...")
                        # Create cache directory if it doesn't exist
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Load data and cache it
                        if experimental_setting.data.voxel_calculation == "all":
                            print(f"\n--------")
                            print(f"- MEAN -")
                            print(f"--------")
                            dataset_dict_mean = load_3d_dataset_with_outer_cv_splits(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation="mean",
                                cv_outer_fold=cv_outer_fold,
                                mode="train",
                                cv_outer_folds_repeats=experimental_setting.cv_outer_folds_repeats,
                                cv_outer_folds_splits=experimental_setting.cv_outer_folds_splits,
                                model_task=experimental_setting.model.task
                            )
                            print(f"\n----------")
                            print(f"- MEDIAN -")
                            print(f"----------")
                            dataset_dict_median = load_3d_dataset_with_outer_cv_splits(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation="median",
                                cv_outer_fold=cv_outer_fold,
                                mode="train",
                                cv_outer_folds_repeats=experimental_setting.cv_outer_folds_repeats,
                                cv_outer_folds_splits=experimental_setting.cv_outer_folds_splits,
                                model_task=experimental_setting.model.task
                            )
                            print(f"\n-------------")
                            print(f"- ISOTROPIC -")
                            print(f"-------------")
                            dataset_dict_isotropic = load_3d_dataset_with_outer_cv_splits(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation="isotropic",
                                cv_outer_fold=cv_outer_fold,
                                mode="train",
                                cv_outer_folds_repeats=experimental_setting.cv_outer_folds_repeats,
                                cv_outer_folds_splits=experimental_setting.cv_outer_folds_splits,
                                model_task=experimental_setting.model.task
                            )
                            print(f"\n------------------------")
                            print(f"- VOLUMETRIC ISOTROPIC -")
                            print(f"------------------------")
                            dataset_dict_volumetric_isotropic = load_3d_dataset_with_outer_cv_splits(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation="volumetric_isotropic",
                                cv_outer_fold=cv_outer_fold,
                                mode="train",
                                cv_outer_folds_repeats=experimental_setting.cv_outer_folds_repeats,
                                cv_outer_folds_splits=experimental_setting.cv_outer_folds_splits,
                                model_task=experimental_setting.model.task
                            )
                            num_classes = dataset_dict_mean["num_classes"]
                            dataset_dict = {
                                "dataset_dict_mean": dataset_dict_mean,
                                "dataset_dict_median": dataset_dict_median,
                                "dataset_dict_isotropic": dataset_dict_isotropic,
                                "dataset_dict_volumetric_isotropic": dataset_dict_volumetric_isotropic,
                            }
                        else:
                            dataset_dict = load_3d_dataset_with_outer_cv_splits(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation=experimental_setting.data.voxel_calculation,
                                cv_outer_fold=cv_outer_fold,
                                mode="train",
                                cv_outer_folds_repeats=experimental_setting.cv_outer_folds_repeats,
                                cv_outer_folds_splits=experimental_setting.cv_outer_folds_splits,
                                model_task=experimental_setting.model.task
                            )
                            num_classes = dataset_dict["num_classes"]
                        
                        # Cache the loaded data
                        cache_data = {
                            "dataset_dict": dataset_dict,
                            "num_classes": num_classes,
                            "cv_outer_fold": cv_outer_fold,
                            "voxel_calculation": experimental_setting.data.voxel_calculation,
                            "dataset": experimental_setting.data.dataset,
                            "seed": experimental_setting.seed
                        }
                        with open(cache_file, "wb") as f:
                            pickle.dump(cache_data, f)
                        print(f"\n\n> 3D data cached to: {cache_file}")
                else:
                    raise NotImplementedError("Cross-validation for 3D datasets is only implemented for cache_data=True.")
            else:
                raise ValueError(f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'")

            print(f"\n{'=' * 100}")
            print(f"Dataset '{experimental_setting.data.dataset}' loaded with {num_classes} classes for outer cross-validation fold {cv_outer_fold + 1}/{cv_outer_folds}")
            print(f"{'=' * 100}\n")
        
        # Mark outer fold as in progress
        status_logger.main_status['outer_folds_progress'][cv_outer_fold + 1] = {
            'status': 'in_progress',                                # Current fold is now running
            'inner_folds_completed': 0,                             # No inner folds completed yet
            'total_inner_folds': total_inner_folds  # Total inner folds for this outer fold
        }
        # Save status for webapp
        status_logger._save_main_status()
        
        # Run NePS optimization for current CV fold
        logging.basicConfig(level=logging.INFO)

        # Create optimizer
        if experimental_setting.searcher == "random_search":  # TODO @Diane: Use user priors or not?
            # For using Random Search with user priors:
            # https://github.com/automl/neps/blob/master/docs/reference/optimizers.md
            # https://automl.github.io/neps/master/api/neps/optimizers/algorithms/?h=true#neps.optimizers.algorithms.random_search
            # optimizer = ("random_search", {"use_priors": True, "ignore_fidelity": True})
            # use_multifidelity = not optimizer[1]["ignore_fidelity"]

            # For using default random search:
            optimizer = experimental_setting.searcher
            use_multifidelity = False
            
        elif experimental_setting.searcher == "priorband":
            # https://automl.github.io/neps/master/api/neps/optimizers/algorithms/?h=true#neps.optimizers.algorithms.priorband
            # Priorband is a multifidelity algorithm, so use_multifidelity = True
            optimizer = ("priorband", {
                "eta": 3,                       # Reduction factor for building brackets (default: 3)
                "sample_prior_first": False,    # Whether to sample the prior configuration first
                "base": "hyperband",            # Base algorithm: "successive_halving", "hyperband", "asha", or "async_hb" (default: "hyperband")
                "bayesian_optimization_kick_in_point": None,  # When to switch to BO (None = disabled) -> int / float / (None = disabled)
            })
            use_multifidelity = True
            
        elif experimental_setting.searcher == "ifbo":
            # https://automl.github.io/neps/master/api/neps/optimizers/algorithms/?h=true#neps.optimizers.algorithms.ifbo
            # IFBO (Iterative Fidelity Bayesian Optimization) is a multifidelity algorithm
            optimizer = ("ifbo", {
                "step_size": 1,                 # Size of the step to take in the fidelity domain (default: 1)
                "use_priors": False,            # Whether to use priors (default: False)
                "sample_prior_first": False,    # Whether to sample the default configuration first (default: False)
                "initial_design_size": "ndim",  # Number of configs to sample before starting optimization (default: "ndim")
                "device": None,                 # Device to use for the model (default: None)
                "surrogate_path": None,         # Path to the surrogate model to use (default: None)
                "surrogate_version": "0.0.1",   # Version of the surrogate model to use (default: "0.0.1")
            })
            use_multifidelity = True
        else:
            raise ValueError(f"Unsupported searcher: {experimental_setting.searcher}. Must be one of: 'random_search', 'priorband', or 'ifbo'. Please integrate your searcher in the code.")
            
        run(
            pipeline_space=pipeline_space,  # Hyperparameter search space
            evaluate_pipeline=lambda pipeline_directory, previous_pipeline_directory, **kwargs: run_pipeline(
                pipeline_directory=pipeline_directory,
                previous_pipeline_directory=previous_pipeline_directory,
                experimental_setting=experimental_setting,
                dataset_dict=dataset_dict,
                num_classes=num_classes,
                use_multifidelity=use_multifidelity,
                **kwargs,
            ),
            optimizer=optimizer,  # HPO algorithm
            root_directory=f"{experimental_setting.neps_directory}/cv_outer_fold_{cv_outer_fold}",
            # max_evaluations_total=experimental_setting.max_evaluations,
            ignore_errors=True,
            cost_to_spend=experimental_setting.cost_to_spend
            # max_cost_total=60,  # e.g., if one config evaluation carries a cost of 2, we can evaluate 5 configs
            # NOTE: In objective_function_3d.py, cost is defined as the epoch time in seconds.
            # We can think about some estimation like: max_cost_total = max_evaluations_total * max_epochs * max_cost_per_epoch
            # TODO @Diane: Define and test max_cost_total
        )
        
        # Update cost.csv after NePS run completes for this outer fold
        # NePS creates report.yaml files after each config evaluation
        # Update CSV in the main NePS output directory (not per outer fold)
        update_cost_csv_from_neps_output(experimental_setting.neps_directory)
        
        # Update outer fold status to completed and mark all inner folds as done
        status_logger.update_neps_progress(
            outer_fold=cv_outer_fold + 1,                                   # Convert to 1-based indexing
            inner_folds_completed=total_inner_folds,  # All inner folds are done
            total_inner_folds=total_inner_folds       # Total inner folds for this outer fold
        )
        
        # Note: Inner fold progress is tracked by InnerFoldProgressLogger in the pipeline
        # This ExperimentStatusLogger only tracks the main NePS status
        
        # Save updated status for webapp
        status_logger._save_main_status()
    
    print(f"\n=== All {cv_outer_folds} Cross-Validation folds completed! ===\n")
    
    # This sets the 'finished' timestamp in neps_status.txt and the webapp will display "Completed" status.
    status_logger.mark_neps_finished()
    
    # Save cross-validation summary to text file
    save_cv_summary(experimental_setting, cv_outer_folds)

    # Automatically summarize evaluation results including outer fold ensemble
    print(f"\n{'='*100}")
    print(f"AUTOMATICALLY SUMMARIZING EVALUATION RESULTS")
    print(f"{'='*100}\n")
    
    try:
        
        # Generate summary with outer fold ensemble
        # Remove seed_XX from experiment_base_dir to get the correct path
        experiment_path = experimental_setting.experiment_base_dir.replace(f"/seed_{experimental_setting.seed}", "")
        summary = summarize_experiment(experiment_path, str(experimental_setting.seed))
        
        # Save summary to file
        summary_file = os.path.join(experimental_setting.experiment_base_dir, "evaluation_summary_across_outer_folds.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\nEvaluation summary saved to: {summary_file}")
        print("\n" + summary)
        
    except Exception as e:
        print(f"Warning: Could not generate evaluation summary: {e}")
        print("You can manually run: python src/analysis/summarize_evaluation_results.py <experiment_path>")

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
