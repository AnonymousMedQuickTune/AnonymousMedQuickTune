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

# For debugging purposes:
# from qtt.predictors import PerfPredictor, CostPredictor
# from external.quicktunetool.src.qtt.predictors import PerfPredictor, CostPredictor
# from external.quicktunetool.src.qtt.optimizers import QuickOptimizer
# from external.quicktunetool.src.qtt.tuners import QuickTuner
# from external.quicktunetool.src.qtt.tuners.image.classification.tuner import QuickImageCLSTuner
# from external.quicktunetool.src.qtt.utils.pretrained import get_pretrained_optimizer


def quicktune_wrapper(trial: dict, trial_info: dict, config: DictConfig) -> dict:
    """
    Wrapper function to adapt run_2d_pipeline for QuickTune's interface.
    """
    start_time = time.time()

    # Prepare directories
    pipeline_dir = Path(trial_info["output-dir"]) / str(trial["config-id"])
    prev_pipeline_dir = None

    # Load dataset
    # TODO: fix hardcoding
    data_dir = Path(trial_info["data-dir"]).parent
    dataset_dict = load_brain_tumor_dataset(data_dir)
    # num_classes = 2

    try:
        # TODO: fix hardcoding
        number_of_epochs = 1  # TODO: trial["fidelity"]

        hyperparameters = trial["config"]
        hyperparameters["number_of_epochs"] = 2  # Add number of epochs to hyperparameters
        print("\n\nHyperparameters: ", hyperparameters, "\n\n")
        

        dimensionality = config.data.dimensionality.lower()

        if dimensionality == "2d":
            result = run_2d_pipeline(
                pipeline_directory=pipeline_dir,
                previous_pipeline_directory=prev_pipeline_dir,
                config=config,
                dataset_dict=dataset_dict,
                num_classes=trial_info["num-classes"],
                **hyperparameters,
            )
        elif dimensionality == "3d":
            # TODO: add model selection (see update in run_2d_pipeline)
            result = run_3d_pipeline(  
                pipeline_directory=pipeline_dir,
                previous_pipeline_directory=prev_pipeline_dir,
                config=config,
                dataset_dict=dataset_dict,
                num_classes=trial_info["num-classes"],
                **hyperparameters,
            )
        else:
            raise ValueError(
                f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
            )
        # Extract metrics safely
        info_dict = result.get("extra", {})
        final_metrics = info_dict.get("all_folds_final_metrics", {})

        score = final_metrics.get(config.metric, 0.0)
        if score is None:
            score = 0.0

        if isinstance(score, (int, float)) and score <= 1.0:
            score = score * 100

        return {
            "config-id": trial["config-id"],
            "status": "SUCCESS",
            "score": score,
            "cost": time.time() - start_time,
            "fidelity": number_of_epochs,
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
            "fidelity": number_of_epochs if "number_of_epochs" in locals() else 4,
            "config": hyperparameters,
        }


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main_experiment_config.yaml",
)
def main(config: DictConfig) -> None:
    """
    Main entry point for the QuickTune training script.

    Args:
        config (DictConfig): Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(config.seed)

    # Create directory for configuration files and logs
    exp_base_dir = Path(config.experiment_base_dir)
    output_dir = exp_base_dir / "hydra_output"
    for directory in [exp_base_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Load and create configuration space
    try:
        with open(config.pipeline_space, "r", encoding="utf-8") as f:
            pipeline_space = yaml.safe_load(f)
        configspace = ConfigSpaceBuilder.from_yaml(pipeline_space)
    except (yaml.YAMLError, IOError) as e:
        logging.error(f"Failed to load pipeline space configuration: {e}")
        raise

    # Prepare and save configurations
    config_files = [
        ("config.yaml", OmegaConf.to_yaml(config)),
        ("pipeline_space.yaml", yaml.dump(pipeline_space, default_flow_style=False)),
    ]
    save_config_files(output_dir, config_files)

    # Create optimizer
    if config.qt.use_medical_portfolio:
        print("\nUse medical Portfolio\n")
        # Load portfolio data
        portfolio = PortfolioManager.load(config.portfolio_dir)
        
        # Extract unique model types from the portfolio
        model_types = portfolio.pipeline_df['model_type'].unique().tolist()
        print(f"\nAvailable models in portfolio: {model_types}\n")
        
        # Add model as a categorical hyperparameter to the configspace
        model_param = CategoricalHyperparameter(
            name="model",
            choices=model_types
        )
        configspace.add(model_param)

        # Merge pipeline configurations with their corresponding metadata
        # Note: Each dataset must have exactly one metadata entry to maintain data integrity
        # and ensure merged_df has the same number of rows as curve/cost data
        merged_df = pd.merge(
            portfolio.pipeline_df, 
            portfolio.meta_df.drop_duplicates(subset=['dataset']),
            how='left',  # Left join ensures:
                        # 1. All pipeline configurations are preserved
                        # 2. Metadata is added only for matching datasets
                        # 3. NaN values for datasets without metadata
                        # This maintains all pipeline configurations for training
        )
        # Remove dataset identifier as it's no longer needed
        merged_df = merged_df.drop(columns=["dataset"])

        # Remove number_of_epochs from merged_df as it's not a hyperparameter and will be set later
        # Number_of_epochs is a fidelity parameter that was needed in the configspace for NePS
        merged_df = merged_df.drop(columns=["number_of_epochs"])

        # Convert learning curves and cost data to numpy arrays for model training
        curve = portfolio.curve_df.values
        cost = portfolio.cost_df.values

        # TODO: check parameters like learning_rate_init, batchsize for predictors
        # TODO: Update batchsize parameter in @CustomCostPredictor and @CustomPerfPredictor when portfolio is ready
        # Note: batchsize is set to 1 for now to avoid division by zero for small portfolio
        # Create predictors with proper paths in experiment_base_dir
        predictor_path = Path(config.experiment_base_dir) / "predictors"
        predictor_path.mkdir(parents=True, exist_ok=True)

        if config.qt.use_ftpfn_perf_predictor:
            # Use FT-PFN performance predictor like in IfBO
            print("\nUse FT-PFN performance predictor\n")
            perf_predictor = FTPFNPerfPredictor(
                path=str(predictor_path / "ftpfn_medical_perf_predictor"),
                seed=config.seed
            ).fit(X=merged_df, y=curve)
        else:
            # Use Quicktune's default SurrogateModel that contains GPRegressionModel
            print("\nUse Quicktune's default SurrogateModel that contains GPRegressionModel\n")
            perf_predictor = CustomPerfPredictor(
                path=str(predictor_path / "medical_perf_predictor"),
                seed=config.seed
            ).fit(X=merged_df, y=curve)

        cost_predictor = CustomCostPredictor(
            path=str(predictor_path / "medical_cost_predictor"),
            seed=config.seed
        ).fit(X=merged_df, y=cost)

        # Save predictors (no need to reset paths anymore)
        perf_predictor.save(verbose=True)
        cost_predictor.save(verbose=True)

        # Initialize optimizer
        optimizer = QuickOptimizer(
            cs=configspace,
            max_fidelity=50,
            cost_aware=True,
            path=config.experiment_base_dir,
            perf_predictor=perf_predictor,
            cost_predictor=cost_predictor,
            seed=config.seed,
        )
    else:
        # Use default MetaAlbum implementation
        print("\nUse default Metaalbum\n")
        optimizer = get_pretrained_optimizer("mtlbm/full")

    print("\nOptimizer created\n")

    # Extract trial info and meta-features from the dataset directory
    # trial_info contains dataset directory and split information,
    # while metafeat contains metadata like number of samples, classes, features and channels.
    # Note: QuickTune's default extract_image_dataset_metafeat() not working for all datasets:
    # https://github.com/automl/quicktunetool/blob/main/src/qtt/finetune/image/classification/utils.py
    # Therefore we use our custom implementation. It's currently hardcoded for the brain_tumor dataset.
    if config.data.dataset == "brain_tumor":
        # TODO: update custom_extract_image_dataset_metafeat() to work for all datasets
        trial_info, metafeat = custom_extract_image_dataset_metafeat( 
            path_root=Path(config.data.path) / "brain_tumor",
            train_split="train",  # Note: overwrite if train / val split is provided   
            val_split="val",  # Note: overwrite if train / val split is provided
        )
    else:
        # TODO: Update custom_extract_image_dataset_metafeat() to work for all our 3D datasets
        raise NotImplementedError(f"Unknown dataset: {config.data.dataset}")

    # Add output directory to trial info
    print("\nTrial info:\n", trial_info, "\n")
    print("\nMeta-features:\n", metafeat, "\n")

    # Setup optimizer with target dataset
    n_of_configs_to_create = 128
    optimizer.setup(n=n_of_configs_to_create, metafeat=metafeat)
    # TODO: check deeper how metafeat is used in optimizer.setup() + maybe add class distribution?

    print("\nOptimizer setup completed\n")

    # Create and run tuner instance - either QuickImageCLSTuner for image classification
    # or standard QuickTuner for general optimization tasks
    if config.qt.use_quick_image_cls_tuner:
        print("\nUse QuickImageCLSTuner\n")
        data_path = Path(config.data.path) / config.data.dataset
        # Using CustomQuickImageCLSTuner with our medical portfolio optimizer
        # instead of default QuickImageCLSTuner which uses MetaAlbum pretrained model
        tuner = CustomQuickImageCLSTuner(
            data_path=str(data_path),  # QuickImageCLSTuner expects string path
            optimizer=optimizer,  # Use our CustomQuickOptimizer
            path=config.experiment_base_dir,
            n=n_of_configs_to_create
        )
        if config.qt.use_custom_objective:
            print("\nUse custom objective\n")
            # Replace the default objective function with our custom one
            
            # Create wrapper that tracks configurations
            def objective_wrapper(trial, trial_info):
                # Get new configuration from optimizer
                config_id = trial.get("config-id", 0)
                print(f"\nSampling configuration {config_id} from optimizer")

                # TODO: check if this is needed
                optimizer.ante()

                # Get configuration from optimizer
                configuration = optimizer.ask()
                print(f"\nConfiguration {config_id}: {configuration}")

                # Run quicktune_wrapper with configuration
                result = quicktune_wrapper(configuration, trial_info, config)

                # Tell the optimizer about the result
                if result is not None:
                    optimizer.tell(result)
                else:
                    print(f"\nTrial {config_id} failed")
                    optimizer.tell(float("-inf"))

                return result
                
            # Use the wrapped objective
            tuner.f = objective_wrapper    
            tuner.run(fevals=config.max_evaluations, trial_info=trial_info)
        else:
            print("\nUse default objective\n")
            tuner.run(fevals=config.max_evaluations)
        
    else:
        print("\nUse QuickTuner\n")
        tuner = QuickTuner(
            optimizer=optimizer,
            f=lambda trial, trial_info: quicktune_wrapper(optimizer.ask(), trial_info, config),
            path=config.experiment_base_dir,
        )
        tuner.run(fevals=config.max_evaluations, time_budget=None, trial_info=trial_info)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
