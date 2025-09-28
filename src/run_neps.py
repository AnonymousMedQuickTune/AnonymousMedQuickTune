import logging
import os
import pickle
from pathlib import Path
import warnings

import hydra
import json
import yaml
from neps import run
from omegaconf import DictConfig, OmegaConf

# Suppress multiprocessing cleanup warnings
warnings.filterwarnings("ignore", message=".*Directory not empty.*")

from src.classification_2d.objective_function_2d import run_2d_pipeline
from src.classification_2d.preprocess_data_2d import load_brain_tumor_dataset
from src.classification_3d.objective_function_3d import run_3d_pipeline
from src.classification_3d.preprocess_data_3d import load_3d_dataset
from src.utils.common_utils import (get_cache_file_path, neps_space_to_dict, set_seed,
                                    yaml_to_neps_pipeline_space, cleanup_training_artifacts)
from src.utils.experiment_status_logger import ExperimentStatusLogger
from src.evaluate_trained_config import evaluate_config_on_test_set
import datetime


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
    # Extract config number from pipeline directory to print in the console
    pipeline_dir_str = str(pipeline_directory)
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
    
    # Evaluate the trained configuration on test set
    print(f"\n{'='*100}")
    print(f"STARTING TEST SET EVALUATION FOR CURRENT CONFIG")
    print(f"{'='*100}\n")
    
    # Extract CV fold from pipeline directory path
    cv_outer_fold = 0  # Default to 0 if not found in path
    if "cv_outer_fold_" in pipeline_dir_str:
        try:
            cv_outer_fold = int(pipeline_dir_str.split("cv_outer_fold_")[-1].split("/")[0])
        except (ValueError, IndexError):
            cv_outer_fold = 0
    
    # Evaluate configuration on test set
    test_metrics = evaluate_config_on_test_set(
        pipeline_directory=pipeline_directory,
        experimental_setting=experimental_setting,
        dataset_dict=dataset_dict,
        num_classes=num_classes,
        hyperparameters=hyperparameters,
        cv_outer_fold=cv_outer_fold
    )

    # Persist only the test metrics as a JSON artifact; do not modify pipeline_result or report.yaml
    if test_metrics is not None:
        test_metrics_file = os.path.join(pipeline_directory, "test_evaluation_results.json")
        with open(test_metrics_file, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=4)
        print(f"Test evaluation completed and saved to: {test_metrics_file}")
    else:
        print(f"Warning: Test evaluation failed or no valid checkpoints found!")
    
    # Delete model checkpoints to save disk space  # TODO @Diane: Keep incumbent model checkpoint?
    # NOTE: After test evaluation, model checkpoints are no longer needed
    cleanup_training_artifacts(pipeline_directory, experimental_setting.cv_inner_folds)
    
    # Print pipeline result and test metrics
    print(f"\n\nPipeline result: {pipeline_result}")
    print(f"\nTest metrics: {test_metrics}\n\n")
    
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

    if experimental_setting.developer_mode:
        print(f"\n\n\nDeveloper mode is enabled!\n\n\n")
        experimental_setting.max_evaluations = 2
        experimental_setting.cv_inner_folds = 2
        experimental_setting.pipeline_space = "configs/pipeline_spaces/pipeline_space_developer_mode.yaml"  # TODO @Diane: Update this
        experimental_setting.training.number_of_epochs = 3
        experimental_setting.cv_outer_folds = 2
    
    if experimental_setting.data.no_validation and not "baseline" in str(experimental_setting.pipeline_space):
        # TODO @Diane: Implement script that takes the best config of a NePS run and retrains it with no validation set
        # NOTE: Update this if after the script is implemented
        raise ValueError("'No validation set' mode is only supported for baseline pipeline space.")

    # Convert YAML pipeline space configuration into NePS-compatible format
    # NePS requires a specific dictionary structure for hyperparameter definitions
    pipeline_space = yaml_to_neps_pipeline_space(experimental_setting.pipeline_space)

    # Print experimental setting and pipeline space
    print("\nexperimental setting: ", experimental_setting, "\n\npipeline space: ", pipeline_space, "\n")

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

    # Cross-validation outer loop for different train+val/test splits
    cv_outer_folds = experimental_setting.cv_outer_folds
    
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
        # Load data for current CV fold
        if experimental_setting.data.use_smart_preprocessing:
            print(f"\n{'=' * 100}")
            print(f"Preloading data for outer cross-validation fold {cv_outer_fold + 1}/{cv_outer_folds} based on the selected voxel calculation method")
            print(f"{'=' * 100}\n")
            # Reload data with current CV fold
            dimensionality = experimental_setting.data.dimensionality.lower()
            if dimensionality == "2d":
                if experimental_setting.data.cache_data:
                    # TODO @Diane: Implement cross-validation for 2D datasets for cache_data=True
                    raise NotImplementedError("Cross-validation for 2D datasets is not implemented yet for cache_data=True.")
                else:
                    # TODO @Diane: Implement cross-validation for 2D datasets
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
                        print(f"> No cache found for outer cross-validation fold {cv_outer_fold + 1}/{cv_outer_folds}. Loading 3D data directly...\n")
                        # Create cache directory if it doesn't exist
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Load data and cache it
                        if experimental_setting.data.voxel_calculation == "all":
                            print(f"--------")
                            print(f"- MEAN -")
                            print(f"--------")
                            dataset_dict_mean = load_3d_dataset(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation="mean",
                                cv_outer_fold=cv_outer_fold,
                                mode="train"
                            )
                            print(f"----------")
                            print(f"- MEDIAN -")
                            print(f"----------")
                            dataset_dict_median = load_3d_dataset(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation="median",
                                cv_outer_fold=cv_outer_fold,
                                mode="train"
                            )
                            print(f"-------------")
                            print(f"- ISOTROPIC -")
                            print(f"-------------")
                            dataset_dict_isotropic = load_3d_dataset(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation="isotropic",
                                cv_outer_fold=cv_outer_fold,
                                mode="train"
                            )
                            print(f"------------------------")
                            print(f"- VOLUMETRIC ISOTROPIC -")
                            print(f"------------------------")
                            dataset_dict_volumetric_isotropic = load_3d_dataset(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation="volumetric_isotropic",
                                cv_outer_fold=cv_outer_fold,
                                mode="train"
                            )
                            num_classes = dataset_dict_mean["num_classes"]
                            dataset_dict = {
                                "dataset_dict_mean": dataset_dict_mean,
                                "dataset_dict_median": dataset_dict_median,
                                "dataset_dict_isotropic": dataset_dict_isotropic,
                                "dataset_dict_volumetric_isotropic": dataset_dict_volumetric_isotropic,
                            }
                        else:
                            dataset_dict = load_3d_dataset(
                                experimental_setting.experiment_base_dir,
                                experimental_setting.data.dataset,
                                data_path=experimental_setting.data.path, 
                                seed=experimental_setting.seed,
                                use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                                voxel_calculation=experimental_setting.data.voxel_calculation,
                                cv_outer_fold=cv_outer_fold,
                                mode="train"
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
            'total_inner_folds': experimental_setting.cv_inner_folds  # Total inner folds for this outer fold
        }
        # Save status for webapp
        status_logger._save_main_status()
        
        # Run NePS optimization for current CV fold
        logging.basicConfig(level=logging.INFO)
        run(
            pipeline_space=pipeline_space,  # Hyperparameter search space
            evaluate_pipeline=lambda pipeline_directory, previous_pipeline_directory, **kwargs: run_pipeline(
                pipeline_directory=pipeline_directory,
                previous_pipeline_directory=previous_pipeline_directory,
                experimental_setting=experimental_setting,
                dataset_dict=dataset_dict,
                num_classes=num_classes,
                **kwargs,
            ),
            optimizer=experimental_setting.searcher,  # HPO algorithm
            root_directory=f"{experimental_setting.neps_directory}/cv_outer_fold_{cv_outer_fold}",
            max_evaluations_total=(
                1 if "baseline" in str(experimental_setting.pipeline_space) else experimental_setting.max_evaluations
            ),
            overwrite_working_directory=False,
            ignore_errors=True,
            # max_cost_total=10,  # e.g., if one config evaluation carries a cost of 2, we can evaluate 5 configs
            # NOTE: In objective_function_3d.py, cost is defined as the epoch time in seconds.
            # We can think about some estimation like: max_cost_total = max_evaluations_total * max_epochs * max_cost_per_epoch
            # TODO @Diane: Define max_cost_total
        )
        
        # Update outer fold status to completed and mark all inner folds as done
        status_logger.update_neps_progress(
            outer_fold=cv_outer_fold + 1,                                   # Convert to 1-based indexing
            inner_folds_completed=experimental_setting.cv_inner_folds,  # All inner folds are done
            total_inner_folds=experimental_setting.cv_inner_folds       # Total inner folds for this outer fold
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


def save_cv_summary(experimental_setting, cv_outer_folds):
    """
    Save cross-validation summary to a text file.
    
    Args:
        experimental_setting (DictConfig): Hydra configuration object
        cv_outer_folds (int): Number of cross-validation folds
    """
    # Create summary directory
    summary_dir = os.path.join(experimental_setting.experiment_base_dir, "cv_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create summary file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(summary_dir, f"cv_summary_{timestamp}.txt")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Experiment information
        f.write("EXPERIMENT INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {experimental_setting.data.dataset}\n")
        f.write(f"Dimensionality: {experimental_setting.data.dimensionality}\n")
        f.write(f"Voxel Calculation: {experimental_setting.data.voxel_calculation}\n")
        f.write(f"Number of Outer Cross-Validation Folds: {cv_outer_folds}\n")
        f.write(f"Seed: {experimental_setting.seed}\n")
        f.write(f"Max Evaluations: {experimental_setting.max_evaluations}\n")
        f.write(f"Optimizer: {experimental_setting.searcher}\n")
        f.write(f"Developer Mode: {experimental_setting.developer_mode}\n")
        f.write(f"Number of Epochs: {experimental_setting.training.number_of_epochs}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # CV Fold directories
        f.write("CROSS-VALIDATION FOLD DIRECTORIES:\n")
        f.write("-" * 40 + "\n")
        for cv_outer_fold in range(cv_outer_folds):
            cv_dir = f"{experimental_setting.neps_directory}/cv_outer_fold_{cv_outer_fold}"
            f.write(f"CV Fold {cv_outer_fold}: {cv_dir}\n")
        f.write("\n")
        
        # Configuration files
        f.write("CONFIGURATION FILES:\n")
        f.write("-" * 40 + "\n")
        config_dir = os.path.join(experimental_setting.experiment_base_dir, "hydra_output")
        f.write(f"Configuration Directory: {config_dir}\n")
        f.write("Files:\n")
        f.write("  - experimental_setting.yaml\n")
        f.write("  - pipeline_space.yaml\n")
        f.write("  - pipeline_space_compact.yaml\n\n")
        
        # Data information
        f.write("DATA INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Data Path: {experimental_setting.data.path}\n")
        f.write(f"Cache Data: {experimental_setting.data.cache_data}\n")
        f.write(f"Use Smart Preprocessing: {experimental_setting.data.use_smart_preprocessing}\n")
        f.write(f"K-Folds: {experimental_setting.cv_inner_folds}\n")
        f.write(f"Num Workers: {experimental_setting.data.num_workers}\n\n")
        
        # Pipeline space information
        f.write("PIPELINE SPACE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pipeline Space File: {experimental_setting.pipeline_space}\n")
        f.write(f"Developer Mode Pipeline: {experimental_setting.developer_mode}\n\n")
        
        # Summary
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total NePS Runs: {cv_outer_folds}\n")
        f.write(f"Each run uses different train+val/test split\n")
        f.write(f"Results saved in separate directories per fold\n")
        f.write(f"Cross-validation ensures robust evaluation\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nCross-validation summary saved to: {summary_file}")
    return summary_file


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
