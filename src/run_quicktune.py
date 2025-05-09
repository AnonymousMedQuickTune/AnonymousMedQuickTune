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
from src.utils.quicktune_utils import custom_extract_image_dataset_metafeat, CustomCostPredictor, CustomPerfPredictor, PortfolioManager, ConfigSpaceBuilder

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
    num_classes = 2

    try:
        # TODO: fix hardcoding
        number_of_epochs = 1  # TODO: trial["fidelity"]
        print("\n\nTrial: ", trial, "\n\n")

        hyperparameters = trial["config"]
        print("\n\n Hyperparameters: ", hyperparameters, "\n\n")

        dimensionality = config.data.dimensionality.lower()

        if dimensionality == "2d":
            result = run_2d_pipeline(
                pipeline_directory=pipeline_dir,
                previous_pipeline_directory=prev_pipeline_dir,
                config=config,
                dataset_dict=dataset_dict,
                num_classes=num_classes,
                **hyperparameters,
            )
        elif dimensionality == "3d":
            # TODO: add model selection (see update in run_2d_pipeline)
            result = run_3d_pipeline(  
                pipeline_directory=pipeline_dir,
                previous_pipeline_directory=prev_pipeline_dir,
                config=config,
                dataset_dict=dataset_dict,
                num_classes=num_classes,
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
    # Set seed for reproducibility
    set_seed(config.seed)

    # Create quicktune output directory with full path structure
    Path(config.experiment_base_dir).mkdir(parents=True, exist_ok=True)

    # Load original pipeline space configuration
    with open(config.pipeline_space, "r", encoding="utf-8") as f:
        pipeline_space = yaml.safe_load(f)

    # Create ConfigurationSpace directly from the original YAML
    configspace = ConfigSpaceBuilder.from_yaml(pipeline_space)

    # Print main experiment configuration and pipeline space
    print("\nconfig: ", config, "\nconfigspace: ", configspace, "\n")

    # Create directory for configuration files and logs
    output_dir = Path(config.experiment_base_dir) / "hydra_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configurations
    config_files = [
        ("config.yaml", OmegaConf.to_yaml(config)),
        ("pipeline_space.yaml", yaml.dump(pipeline_space, default_flow_style=False)),
    ]

    for filename, data in config_files:
        config_path = output_dir / filename
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(data)
        except IOError as e:
            logging.error(f"Failed to write configuration file {filename}: {e}")

    if config.qt.use_medical_portfolio:
        print("\n\nUse medical Portfolio\n\n")
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

        # Convert learning curves and cost data to numpy arrays for model training
        curve = portfolio.curve_df.values
        cost = portfolio.cost_df.values

        # TODO: check parameters like learning_rate_init, batchsize for predictors
        # TODO: Update batchsize parameter in @CustomCostPredictor and @CustomPerfPredictor when portfolio is ready
        # Note: batchsize is set to 1 for now to avoid division by zero for small portfolio
        perf_predictor = CustomPerfPredictor().fit(X=merged_df, y=curve)
        cost_predictor = CustomCostPredictor().fit(X=merged_df, y=cost)

        # Save predictors for later evaluation
        predictor_path = Path(config.experiment_base_dir) / "predictors"
        predictor_path.mkdir(parents=True, exist_ok=True)
        
        perf_predictor.reset_path(str(predictor_path / "perf"))
        cost_predictor.reset_path(str(predictor_path / "cost"))
        
        perf_predictor.save(verbose=True)
        cost_predictor.save(verbose=True)

        # Initialize optimizer with both predictors
        optimizer = QuickOptimizer(
            cs=configspace,
            max_fidelity=50,  # TODO: fix hardcoding
            cost_aware=True,
            path=config.experiment_base_dir,
            perf_predictor=perf_predictor,
            cost_predictor=cost_predictor,
            seed=config.seed,
        )
        # Explicitly reset path for pretrained optimizer
        optimizer.reset_path(config.experiment_base_dir)
    else:
        # Initialize the optimizer with pretrained model
        print("\n\nUse default Metaalbum\n\n")

        raise NotImplementedError("Not implemented. Use medical portfolio instead.")
    
        optimizer = get_pretrained_optimizer("mtlbm/full")
        # Explicitly reset path for pretrained optimizer
        optimizer.reset_path(
            config.experiment_base_dir
        ) 

    print("\nOptimizer created\n")

    # Extract trial info and meta-features from the dataset directory
    # trial_info contains dataset directory and split information,
    # while metafeat contains metadata like number of samples, classes, features and channels.
    # Note: QuickTune's default extract_image_dataset_metafeat() not working for all datasets:
    # https://github.com/automl/quicktunetool/blob/main/src/qtt/finetune/image/classification/utils.py
    # Therefore we use our custom implementation.
    if config.data.dataset == "brain_tumor":
        # TODO: update custom_extract_image_dataset_metafeat() to work for all datasets
        trial_info, metafeat = custom_extract_image_dataset_metafeat( 
            path_root=Path(config.data.path) / "brain_tumor",
            train_split="train",  # Note: overwrite if train / val split is provided   
            val_split="val",  # Note: overwrite if train / val split is provided
        )
    else:
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
        tuner = QuickImageCLSTuner(
            data_path=str(data_path),  # QuickImageCLSTuner expects string path
            path=config.experiment_base_dir,
            n=n_of_configs_to_create
        )
        if config.qt.use_custom_objective:
            print("\nUse custom objective\n")
            # Replace the default objective function with our custom one
            
            # Create wrapper that tracks configurations
            def objective_wrapper(trial, trial_info):
                # Get new configuration from optimizer
                print(f"\nTrial: {trial}")
                config_id = trial.get("config-id", 0)
                print(f"\nSampling configuration {config_id} from optimizer")

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
