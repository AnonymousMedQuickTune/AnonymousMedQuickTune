import json
import logging
import os
import time
import traceback
import copy
from dataclasses import dataclass
from pathlib import Path

import hydra
import pandas as pd
import yaml
from ConfigSpace import (CategoricalHyperparameter, ConfigurationSpace,
                         UniformFloatHyperparameter,
                         UniformIntegerHyperparameter)
from omegaconf import DictConfig, OmegaConf
from qtt import QuickOptimizer, QuickTuner, QuickImageCLSTuner, get_pretrained_optimizer

from src.classification_2d.objective_function_2d import run_2d_pipeline
from src.classification_3d.objective_function_3d import run_3d_pipeline
from src.classification_2d.preprocess_data_2d import load_brain_tumor_dataset
from src.classification_3d.preprocess_data_3d import load_3d_dataset
from src.utils.common_utils import set_seed
from src.utils.quicktune_utils import (
    CustomQuickImageCLSTuner,
    CustomCostPredictor,
    CustomPerfPredictor,
    PortfolioManager,
    ConfigSpaceBuilder,
    save_config_files,
    custom_extract_image_dataset_metafeat,
    FTPFNPerfPredictor,
)
from src.utils.experiment_status_logger import ExperimentStatusLogger, InnerFoldProgressLogger
from src.evaluate_trained_config import evaluate_config_on_test_set
from src.utils.common_utils import cleanup_training_artifacts

# For debugging purposes:
# from qtt.predictors import PerfPredictor, CostPredictor
# from external.quicktunetool.src.qtt.predictors import PerfPredictor, CostPredictor
# from external.quicktunetool.src.qtt.optimizers import QuickOptimizer
# from external.quicktunetool.src.qtt.tuners import QuickTuner
# from external.quicktunetool.src.qtt.tuners.image.classification.tuner import QuickImageCLSTuner
# from external.quicktunetool.src.qtt.utils.pretrained import get_pretrained_optimizer


def quicktune_wrapper(trial: dict, trial_info: dict, experimental_setting: DictConfig, cv_outer_fold: int = 1, status_logger: ExperimentStatusLogger = None) -> dict:
    """
    Wrapper function to adapt run_2d_pipeline and run_3d_pipeline for QuickTune's interface.
    
    This function serves as a bridge between QuickTune's optimization framework and the existing
    NePS-compatible pipeline functions. It handles dataset loading, dimensionality detection,
    pipeline execution, test set evaluation, and result formatting to ensure seamless integration 
    with QuickTune's optimization process.
    
    Args:
        trial (dict): QuickTune trial configuration containing:
            - config-id (int): Unique identifier for this trial
            - config (dict): Hyperparameter configuration to evaluate
            - fidelity (int): Fidelity level for the evaluation (e.g., number of epochs)
        trial_info (dict): Dataset information containing:
            - data-dir (str): Path to the dataset directory
            - train-split (str): Name of the training split directory
            - val-split (str): Name of the validation split directory  
            - num-classes (int): Number of classes in the dataset
        experimental_setting (DictConfig): Hydra configuration object containing:
            - data.dataset (str): Name of the dataset ('lipo', 'desmoid', 'brain_tumor')
            - data.dimensionality (str): Dataset dimensionality ('2d' or '3d')
            - data.voxel_calculation (str): Voxel calculation method for 3D datasets
            - data.use_smart_preprocessing (bool): Whether to use smart preprocessing
            - metric (str): Evaluation metric to optimize
            - seed (int): Random seed for reproducibility
            - etc.
        cv_outer_fold (int, optional): Outer cross-validation fold number. Defaults to 1.
        status_logger (ExperimentStatusLogger, optional): Status logger for tracking experiment progress.
            Used to log inner fold progress and create status files for webapp dashboard.
            If None, no status logging is performed.
    
    Returns:
        dict: Result dictionary containing:
            - config-id (int): Trial identifier
            - status (str): Trial status ('SUCCESS' or 'FAILED')
            - score (float): Performance score (scaled to 0-100 if <= 1.0)
            - cost (float): Evaluation time in seconds  # TODO @Diane: Checkout whether to take time in sec or #epochs (early stopping)
            - fidelity (int): Fidelity level used
            - config (dict): Hyperparameter configuration
    
    Process:
        1. Load dataset based on dimensionality and dataset type
        2. Run the appropriate training pipeline (2D or 3D) with cross-validation
        3. Evaluate the trained configuration on the test set using ensemble predictions
        4. Save test evaluation results to test_evaluation_results.json
        5. Clean up model checkpoints to save disk space
        6. Return formatted result to QuickTune for optimization
    
    Raises:
        ValueError: If unsupported dataset or dimensionality is specified
        Exception: If pipeline execution fails (caught and returned as FAILED status)
    
    Note:
        - Failed trials return -inf score and inf cost to signal optimization failure
        - Status logging creates config-specific status files in experiment_status/config_X/outerfold_Y_status.txt format
        - Test evaluation uses cross-validation ensemble predictions for robust performance assessment
        - Test metrics are saved as JSON files for detailed analysis
    """
    start_time = time.time()

    # Prepare directories
    pipeline_dir = Path(trial_info["output-dir"]) / str(trial["config-id"])
    prev_pipeline_dir = None
    
    # Initialize inner fold progress logger for this configuration
    if status_logger is not None:
        inner_fold_logger = InnerFoldProgressLogger(str(pipeline_dir))

    # Load dataset based on dimensionality
    dimensionality = experimental_setting.data.dimensionality
    
    if dimensionality == "2d":
        if experimental_setting.data.dataset == "brain_tumor":
            data_dir = Path(trial_info["data-dir"]).parent
            dataset_dict = load_brain_tumor_dataset(data_dir)
            num_classes = 2
        else:
            raise ValueError(f"Unsupported 2D dataset: {experimental_setting.data.dataset}")

    elif dimensionality == "3d":
        if experimental_setting.data.dataset in ["lipo", "desmoid"]:

            voxel_calculation = experimental_setting.data.voxel_calculation
            if voxel_calculation == "all":
                # Load all voxel calculation methods like in run_neps.py
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
                    cv_outer_fold=cv_outer_fold,  # Use current CV fold
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
                    cv_outer_fold=cv_outer_fold,  # Use current CV fold
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
                    cv_outer_fold=cv_outer_fold,  # Use current CV fold
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
                    cv_outer_fold=cv_outer_fold,  # Use current CV fold
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
                # Load single voxel calculation method
                dataset_dict = load_3d_dataset(
                    experimental_setting.experiment_base_dir,
                    experimental_setting.data.dataset,
                    data_path=experimental_setting.data.path,
                    seed=experimental_setting.seed,
                    use_smart_preprocessing=experimental_setting.data.use_smart_preprocessing,
                    voxel_calculation=voxel_calculation,
                    cv_outer_fold=cv_outer_fold,  # Use current CV fold
                    mode="train"
                )
                num_classes = dataset_dict["num_classes"]
        else:
            raise ValueError(f"Unsupported 3D dataset: {experimental_setting.data.dataset}")
    else:
        raise ValueError(f"Unsupported dimensionality: {dimensionality}")

    # Print hyperparameters
    hyperparameters = trial["config"]
    print("\n\nHyperparameters: ", hyperparameters, "\n\n")

    # Set fidelity as number of epochs
    trial["fidelity"] = hyperparameters["number_of_epochs"]
    print("trial fidelity (# epochs): ", trial["fidelity"])

    try:
        # Start run pipeline based on dimensionality. Use the dimensionality we determined earlier, don't override it
        print(f"\n{'-' * 100}")
        print("> Start Run Pipeline")
        print(f"{'-' * 100}")

        # Log inner fold progress if status logger is available
        if status_logger is not None:
            inner_fold_logger.update_inner_fold_progress(
                inner_fold=1,  # Start with first inner fold
                status='in_progress',
                total_inner_folds=experimental_setting.cv_inner_folds
            )

        if dimensionality == "2d":
            result = run_2d_pipeline(
                pipeline_directory=pipeline_dir,
                previous_pipeline_directory=prev_pipeline_dir,
                experimental_setting=experimental_setting,
                dataset_dict=dataset_dict,
                num_classes=num_classes,
                inner_fold_logger=inner_fold_logger if status_logger is not None else None,
                **hyperparameters,
            )
        elif dimensionality == "3d":
            result = run_3d_pipeline(  
                pipeline_directory=pipeline_dir,
                previous_pipeline_directory=prev_pipeline_dir,
                experimental_setting=experimental_setting,
                dataset_dict=dataset_dict,
                num_classes=num_classes,
                inner_fold_logger=inner_fold_logger if status_logger is not None else None,
                **hyperparameters,
            )
        else:
            raise ValueError(
                f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
            )
        
        # Inner fold progress is now logged within the pipeline functions

        # Evaluate the trained configuration on test set
        print(f"\n{'='*100}")
        print(f"STARTING TEST SET EVALUATION FOR CURRENT CONFIG")
        print(f"{'='*100}\n")
        
        # Evaluate configuration on test set
        test_metrics = evaluate_config_on_test_set(
            pipeline_directory=str(pipeline_dir),
            experimental_setting=experimental_setting,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            hyperparameters=hyperparameters,
            cv_outer_fold=cv_outer_fold,
            framework="quicktune"
        )

        # Persist test metrics as a JSON artifact
        if test_metrics is not None:
            test_metrics_file = pipeline_dir / "test_evaluation_results.json"
            with open(test_metrics_file, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=4)
            print(f"Test evaluation completed and saved to: {test_metrics_file}")
        else:
            print(f"Warning: Test evaluation failed or no valid checkpoints found!")

        # Delete model checkpoints to save disk space after test evaluation
        cleanup_training_artifacts(str(pipeline_dir), experimental_setting.cv_inner_folds)

        # Extract metrics safely
        info_dict = result.get("extra", {})
        final_metrics = info_dict.get("all_folds_final_metrics", {})

        # TODO @Diane: check if QuickTune is minimizing or maximizing its objective function
        score = final_metrics.get(experimental_setting.metric, 0.0)

        # TODO @Diane: Also log other metrics for QuickTune

        if isinstance(score, (int, float)) and score <= 1.0:
            score = score * 100

        return {
            "config-id": trial["config-id"],
            "status": "SUCCESS",
            "score": score,
            "cost": time.time() - start_time,
            "fidelity": trial["fidelity"],
            "config": hyperparameters,
        }

    except Exception as e:
        print(f"Error in pipeline: {e}")
        traceback.print_exc()
        return {
            "config-id": trial["config-id"],
            "status": "FAILED",
            "score": float("-inf"),
            "cost": float("inf"),
            "fidelity": trial["fidelity"],
            "config": hyperparameters,
        }


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experimental_setting.yaml",
)
def main(experimental_setting: DictConfig) -> None:
    """
    Main entry point for the QuickTune training script.

    Args:
        experimental_setting (DictConfig): Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(experimental_setting.seed)

    # Developer mode (for faster/smaller experiments)
    if experimental_setting.developer_mode:
        print(f"\n\n\nDeveloper mode is enabled!\n\n\n")
        experimental_setting.max_evaluations = 2
        experimental_setting.cv_inner_folds = 2
        experimental_setting.pipeline_space = "configs/pipeline_spaces/pipeline_space_developer_mode.yaml"  # TODO @Diane: Update this
        experimental_setting.training.number_of_epochs = 3
        experimental_setting.cv_outer_folds = 2

    # Create directory for configuration files and logs
    exp_base_dir = Path(experimental_setting.experiment_base_dir)
    output_dir = exp_base_dir / "hydra_output"
    for directory in [exp_base_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Load and create configuration space
    try:
        with open(experimental_setting.pipeline_space, "r", encoding="utf-8") as f:
            pipeline_space = yaml.safe_load(f)
        configspace = ConfigSpaceBuilder.from_yaml(pipeline_space)
    except (yaml.YAMLError, IOError) as e:
        logging.error(f"Failed to load pipeline space configuration: {e}")
        raise

    # Prepare and save configurations
    config_files = [
        ("experimental_setting.yaml", OmegaConf.to_yaml(experimental_setting)),
        ("pipeline_space.yaml", yaml.dump(pipeline_space, default_flow_style=False)),
    ]
    save_config_files(output_dir, config_files)

    # STEP 1:
    # Extract trial_info and metafeat from the dataset directory
    # trial_info contains dataset directory and split information,
    # while metafeat contains metadata like number of samples, classes, features and channels.
    # NOTE: QuickTune's default extract_image_dataset_metafeat() not working for all datasets:
    # https://github.com/automl/quicktunetool/blob/main/src/qtt/finetune/image/classification/utils.py
    # Therefore we use our custom implementation.
    # TODO @Diane: update custom_extract_image_dataset_metafeat() to work for all datasets
    # TODO @Diane: think about adding more metafeatures such as modality, class distribution
    if experimental_setting.data.dataset == "brain_tumor":
        trial_info, metafeat = custom_extract_image_dataset_metafeat( 
            path_root=Path(experimental_setting.data.path) / "brain_tumor",
            train_split="train",  # NOTE: overwrite if train / val split is provided   
            val_split="val",  # NOTE: overwrite if train / val split is provided
        )
    elif experimental_setting.data.dataset in ["lipo", "desmoid"]:  # TODO @Diane: Fix placeholder values!
        # Use portfolio meta-features for 3D datasets
        trial_info, metafeat = custom_extract_image_dataset_metafeat( 
            path_root=Path(experimental_setting.data.path) / experimental_setting.data.dataset,
            train_split="train",  # NOTE: overwrite if train / val split is provided   
            val_split="val",  # NOTE: overwrite if train / val split is provided
        )
    else:
        # TODO @Diane: Update custom_extract_image_dataset_metafeat() to work for all our 3D datasets
        raise NotImplementedError(f"Unknown dataset: {experimental_setting.data.dataset}")

    # Add output directory to trial info
    print("\nTrial info:\n", trial_info, "\n")
    print("\nMeta features:\n", metafeat, "\n")

    # Cross-validation outer loop for different train+val/test splits (like in run_neps.py)
    cv_outer_folds = experimental_setting.cv_outer_folds if hasattr(experimental_setting, 'cv_outer_folds') else 1
    
    # Initialize experiment status logger for QuickTune
    status_logger = ExperimentStatusLogger(experimental_setting.experiment_base_dir, experiment_type="quicktune")
    
    # Set the total number of outer folds for cross-validation to calculate overall progress percentages
    status_logger.set_total_outer_folds(cv_outer_folds)
    
    # Force save the initial status immediately after initialization to ensure the webapp can display "Active" status
    status_logger._save_main_status()
    
    print(f"\n=== Starting Cross-Validation with {cv_outer_folds} folds ===\n")
    
    for cv_outer_fold in range(cv_outer_folds):
        print(f"\n{'=' * 100}")
        print(f"Starting QuickTune optimization for CV fold {cv_outer_fold + 1}/{cv_outer_folds}")
        print(f"{'=' * 100}\n")
        
        # Create experiment directory for this CV fold
        cv_experiment_dir = Path(experimental_setting.experiment_base_dir) / f"cv_outer_fold_{cv_outer_fold}"
        cv_experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Mark outer fold as in progress
        status_logger.main_status['outer_folds_progress'][cv_outer_fold + 1] = {
            'status': 'in_progress',
            'inner_folds_completed': 0,
            'total_inner_folds': experimental_setting.cv_inner_folds
        }
        # Save status for webapp
        status_logger._save_main_status()
        
        # STEP 2: Create a fresh optimizer for each CV fold to ensure independent optimization
        if experimental_setting.qt.use_medical_portfolio:
            print("\nUse medical Portfolio\n")
            # Load portfolio data
            portfolio = PortfolioManager.load(experimental_setting.portfolio_dir)
            
            # Extract unique model types from the portfolio
            model_types = portfolio.pipeline_df['model_type'].unique().tolist()
            print(f"\nAvailable models in portfolio: {model_types}\n")
            
            # Create a fresh configspace for this CV fold
            cv_configspace = ConfigSpaceBuilder.from_yaml(pipeline_space)
            
            # Add model as a categorical hyperparameter to the configspace
            model_param = CategoricalHyperparameter(
                name="model",
                choices=model_types
            )
            cv_configspace.add(model_param)

            # Merge pipeline configurations with their corresponding metadata
            # Note: Each dataset must have exactly one metadata entry to maintain data integrity
            # and ensure merged_df has the same number of rows as curve/cost data
            # Left join ensures:
                # 1. All pipeline configurations are preserved
                # 2. Metadata is added only for matching datasets
                # 3. NaN values for datasets without metadata
                # This maintains all pipeline configurations for training
            merged_df = pd.merge(
                portfolio.pipeline_df, 
                portfolio.meta_df.drop_duplicates(subset=['dataset']),
                how='left',
            )
            # Remove dataset identifier as it's no longer needed
            merged_df = merged_df.drop(columns=["dataset"])

            # Only drop number_of_epochs if it exists in the dataframe as it's not a hyperparameter and will be set later
            # Number_of_epochs is a fidelity parameter that was needed in the configspace for NePS
            if "number_of_epochs" in merged_df.columns:  # TODO @Diane: double check this
                merged_df = merged_df.drop(columns=["number_of_epochs"])

            # Convert learning curves and cost data to numpy arrays for model training
            curve = portfolio.curve_df.values
            cost = portfolio.cost_df.values

            # TODO @Diane: check parameters like learning_rate_init, batchsize for predictors
            # TODO @Diane: Update batchsize parameter in @CustomCostPredictor and @CustomPerfPredictor when portfolio is ready
            # NOTE: batchsize is set to 1 for now to avoid division by zero for small portfolio
            # Create predictors with proper paths in experiment_base_dir
            predictor_path = cv_experiment_dir / "predictors"
            predictor_path.mkdir(parents=True, exist_ok=True)

            if experimental_setting.qt.use_ftpfn_perf_predictor:
                # Use FT-PFN performance predictor like in IfBO
                print("\nUse FT-PFN performance predictor\n")
                perf_predictor = FTPFNPerfPredictor(
                    path=str(predictor_path / "ftpfn_medical_perf_predictor"),
                    seed=experimental_setting.seed
                ).fit(X=merged_df, y=curve)
            else:
                # Use Quicktune's default SurrogateModel that contains GPRegressionModel
                print("\nUse Quicktune's default SurrogateModel that contains GPRegressionModel\n")
                perf_predictor = CustomPerfPredictor(
                    path=str(predictor_path / "medical_perf_predictor"),
                    seed=experimental_setting.seed
                ).fit(X=merged_df, y=curve)

            cost_predictor = CustomCostPredictor(
                path=str(predictor_path / "medical_cost_predictor"),
                seed=experimental_setting.seed
            ).fit(X=merged_df, y=cost)

            # Save predictors
            perf_predictor.save(verbose=True)
            cost_predictor.save(verbose=True)

            # Initialize fresh optimizer for this CV fold
            optimizer = QuickOptimizer(
                cs=cv_configspace,
                max_fidelity=50,
                cost_aware=True,
                path=str(cv_experiment_dir),
                perf_predictor=perf_predictor,
                cost_predictor=cost_predictor,
                seed=experimental_setting.seed,
            )
        else:
            # Use default MetaAlbum implementation
            print("\nUse default Metaalbum\n")
            optimizer = get_pretrained_optimizer("mtlbm/full")

        print("\nOptimizer created\n")
        
        # STEP 3: Setup optimizer with target dataset
        n_of_configs_to_create = 128  # TODO @Diane: Check this out!
        optimizer.setup(n=n_of_configs_to_create, metafeat=metafeat)
        # TODO: check deeper how metafeat is used in optimizer.setup() + maybe add class distribution?
        
        print("\nOptimizer setup completed\n")
        
        # Create and run tuner instance - either QuickImageCLSTuner for image classification
        # or standard QuickTuner for general optimization tasks
        if experimental_setting.qt.use_quick_image_cls_tuner:
            print("\nUse QuickImageCLSTuner\n")
            data_path = Path(experimental_setting.data.path) / experimental_setting.data.dataset
            # Using CustomQuickImageCLSTuner with our medical portfolio optimizer
            # instead of default QuickImageCLSTuner which uses MetaAlbum pretrained model
            tuner = CustomQuickImageCLSTuner(
                data_path=str(data_path),  # QuickImageCLSTuner expects string path
                optimizer=optimizer,  # Use our CustomQuickOptimizer
                path=str(cv_experiment_dir),  # Use CV fold specific directory
                n=n_of_configs_to_create
            )
            if experimental_setting.qt.use_custom_objective:
                print("\nUse custom objective\n")
                # Replace the default objective function with our custom one
                
                # Create wrapper that tracks configurations
                def objective_wrapper(trial, trial_info):
                    # Get new configuration from optimizer
                    config_id = trial.get("config-id", 0)
                    print(f"\nSampling configuration {config_id} from optimizer")

                    # TODO @Diane: check if this is needed
                    optimizer.ante()

                    # Get configuration from optimizer
                    configuration = optimizer.ask()
                    print(f"\nConfiguration {config_id}: {configuration}")

                    # Run quicktune_wrapper with configuration
                    result = quicktune_wrapper(configuration, trial_info, experimental_setting, cv_outer_fold, status_logger)

                    # Tell the optimizer about the result
                    if result is not None:
                        optimizer.tell(result)
                    else:
                        print(f"\nTrial {config_id} failed")
                        optimizer.tell(float("-inf"))

                    return result
                    
                # Use the wrapped objective
                tuner.f = objective_wrapper    
                tuner.run(fevals=experimental_setting.max_evaluations, trial_info=trial_info)
            else:
                print("\nUse default objective\n")
                tuner.run(fevals=experimental_setting.max_evaluations)
            
        else:
            print("\nUse QuickTuner\n")
            tuner = QuickTuner(
                optimizer=optimizer,
                f=lambda trial, trial_info: quicktune_wrapper(optimizer.ask(), trial_info, experimental_setting, cv_outer_fold, status_logger),
                path=str(cv_experiment_dir),
            )
            tuner.run(fevals=experimental_setting.max_evaluations, time_budget=None, trial_info=trial_info)
        
        # Update outer fold status to completed and mark all inner folds as done
        status_logger.update_main_progress(
            outer_fold=cv_outer_fold + 1,                                   # Convert to 1-based indexing
            inner_folds_completed=experimental_setting.cv_inner_folds,  # All inner folds are done
            total_inner_folds=experimental_setting.cv_inner_folds       # Total inner folds for this outer fold
        )
        
        # Save updated status for webapp
        status_logger._save_main_status()
        
        print(f"\n{'=' * 100}")
        print(f"Completed QuickTune optimization for CV fold {cv_outer_fold + 1}/{cv_outer_folds}")
        print(f"{'=' * 100}\n")
    
    # Mark QuickTune experiment as finished
    status_logger.mark_main_finished()
    
    print(f"\n=== All {cv_outer_folds} Cross-Validation folds completed! ===\n")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
