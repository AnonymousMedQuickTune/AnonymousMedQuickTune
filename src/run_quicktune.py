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
from src.classification_2d.preprocess_data_2d import load_brain_tumor_dataset, get_max_batch_size
from src.utils.common_utils import set_seed, yaml_to_neps_pipeline_space
from src.utils.quicktune_utils import custom_extract_image_dataset_metafeat
from qtt.predictors import PerfPredictor, CostPredictor


@dataclass
class PortfolioData:
    """Container for portfolio data files"""

    pipeline_df: pd.DataFrame
    curve_df: pd.DataFrame
    cost_df: pd.DataFrame
    meta_df: pd.DataFrame


class ConfigSpaceBuilder:
    """Handles creation of ConfigurationSpace from YAML"""

    @staticmethod
    def from_yaml(config_dict: dict) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        type_to_param = {
            "float": UniformFloatHyperparameter,
            "int": UniformIntegerHyperparameter,
            "categorical": CategoricalHyperparameter,
        }

        for param_name, param_config in config_dict.items():
            param_type = param_config["type"]
            param_class = type_to_param.get(param_type)

            if not param_class:
                raise ValueError(f"Unknown parameter type: {param_type}")

            # Convert scientific notation strings to float
            if param_type in ["float", "int"]:
                lower = (
                    float(param_config["lower"])
                    if isinstance(param_config["lower"], str)
                    else param_config["lower"]
                )
                upper = (
                    float(param_config["upper"])
                    if isinstance(param_config["upper"], str)
                    else param_config["upper"]
                )

                if param_type == "float":
                    param = param_class(
                        name=param_name,
                        lower=lower,
                        upper=upper,
                        log=param_config.get("log", False),
                    )
                else:  # int
                    param = param_class(
                        name=param_name, lower=int(lower), upper=int(upper)
                    )
            else:  # categorical
                param = param_class(name=param_name, choices=param_config["choices"])

            cs.add_hyperparameter(param)

        return cs


class PortfolioManager:
    """Handles loading and saving of portfolio data"""

    @staticmethod
    def load(portfolio_dir: str) -> PortfolioData:
        """Load portfolio data from CSV files"""
        try:
            return PortfolioData(
                pipeline_df=pd.read_csv(f"{portfolio_dir}/config.csv"),
                curve_df=pd.read_csv(f"{portfolio_dir}/curve.csv"),
                cost_df=pd.read_csv(f"{portfolio_dir}/cost.csv"),
                meta_df=pd.read_csv(f"{portfolio_dir}/meta.csv"),
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Portfolio file not found in {portfolio_dir}: {e}")

    @staticmethod
    def save(portfolio: PortfolioData, output_dir: str):
        """Save portfolio data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        portfolio.pipeline_df.to_csv(f"{output_dir}/config.csv", index=False)
        portfolio.curve_df.to_csv(f"{output_dir}/curve.csv", index=True)
        portfolio.cost_df.to_csv(f"{output_dir}/cost.csv", index=False)
        portfolio.meta_df.to_csv(f"{output_dir}/meta.csv", index=False)


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
    # TODO: Override hydra output directory for QuickTune
    
    # Set seed for reproducibility
    set_seed(config.seed)

    # Create quicktune output directory with full path structure
    Path(config.qt.experiment_base_dir).mkdir(parents=True, exist_ok=True)

    # Load original pipeline space configuration
    with open(config.pipeline_space, "r", encoding="utf-8") as f:
        pipeline_space = yaml.safe_load(f)

    # Create ConfigurationSpace directly from the original YAML
    configspace = ConfigSpaceBuilder.from_yaml(pipeline_space)

    # Print main experiment configuration and pipeline space
    print("\nconfig: ", config, "\nconfigspace: ", configspace, "\n")

    # Create directory for configuration files and logs
    output_dir = Path(config.qt.experiment_base_dir) / "hydra_output"
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
        portfolio = PortfolioManager.load(config.qt.portfolio_dir)
        
        # Extract unique model types from the portfolio
        model_types = portfolio.pipeline_df['model_type'].unique().tolist()
        print(f"\nAvailable models in portfolio: {model_types}\n")
        
        # Add model as a categorical hyperparameter to the configspace
        model_param = CategoricalHyperparameter(
            name="model",
            choices=model_types
        )
        configspace.add_hyperparameter(model_param)

        # TODO: delete prints for debugging (after fixing bug that occurs when using CostPredictor)
        print("\nShape of data before merge:")
        print(f"Pipeline DF shape: {portfolio.pipeline_df.shape}")
        print(f"Meta DF shape: {portfolio.meta_df.shape}")
        print(f"Curve shape: {portfolio.curve_df.shape}")
        print(f"Cost shape: {portfolio.cost_df.shape}\n")

        # Debug dataset values
        print("\nUnique datasets in pipeline_df:", portfolio.pipeline_df['dataset'].unique())
        print("Unique datasets in meta_df:", portfolio.meta_df['dataset'].unique())

        # Get max batch size from pipeline space
        pipeline_space = yaml_to_neps_pipeline_space(config.pipeline_space)
        max_batch_size = get_max_batch_size(pipeline_space)
        print(f"\nUsing max batch size: {max_batch_size}")

        # Prepare data - ensure we don't duplicate rows
        merged_df = pd.merge(
            portfolio.pipeline_df, 
            portfolio.meta_df.drop_duplicates(subset=['dataset']), 
            on="dataset",
            how='left'  # Use left merge to keep pipeline_df rows
        )
        merged_df = merged_df.drop(columns=["dataset"])
        
        # Drop any unnamed columns
        merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
        
        print(f"\nAfter cleaning:")
        print(f"Merged DF shape: {merged_df.shape}")
        print(f"Merged DF columns: {merged_df.columns.tolist()}\n")

        # Convert curve and cost to correct format
        curve = portfolio.curve_df.values
        cost = portfolio.cost_df['cost'].values.reshape(-1, 1)

        print(f"Final shapes:")
        print(f"merged_df: {merged_df.shape}")
        print(f"curve: {curve.shape}")
        print(f"cost: {cost.shape}\n")

        # Define separate fit parameters for perf and cost predictors
        # TODO: check parameters like learning_rate_init for PerfPredictor
        perf_predictor = PerfPredictor().fit(
            X=merged_df, 
            y=curve,
            batch_size=max(1, min(max_batch_size, merged_df.shape[0])),
            epochs=4,
            patience=5,
            validation_fraction=0.1,
            early_stop=True,
            learning_rate_init=0.001
        )
        
        # CostPredictor with identical parameters to PerfPredictor
        # TODO: fix Bug that occurs when using CostPredictor
        # TODO: check parameters like learning_rate_init for CostPredictor
        """
        cost_predictor = CostPredictor().fit(
            X=merged_df, 
            y=cost,
            batch_size=max(1, min(max_batch_size, merged_df.shape[0])),  # Same as PerfPredictor
            epochs=100,
            patience=5,
            validation_fraction=0.1,  # Same as PerfPredictor
            early_stop=True,  # Same as PerfPredictor
            learning_rate_init=0.001
        )
        """

        # Save predictors for later evaluation
        predictor_path = Path(config.qt.experiment_base_dir) / "predictors"
        predictor_path.mkdir(parents=True, exist_ok=True)
        
        perf_predictor.reset_path(str(predictor_path / "perf"))
        # cost_predictor.reset_path(str(predictor_path / "cost"))
        
        perf_predictor.save(verbose=True)
        # cost_predictor.save(verbose=True)

        # Initialize optimizer with both predictors
        optimizer = QuickOptimizer(
            cs=configspace,
            max_fidelity=50,  # TODO: fix hardcoding
            cost_aware=True,
            path=config.qt.experiment_base_dir,
            perf_predictor=perf_predictor,
            # cost_predictor=cost_predictor,
        )
        # Explicitly reset path for pretrained optimizer
        optimizer.reset_path(config.qt.experiment_base_dir)
    else:
        # Initialize the optimizer with pretrained model
        print("\n\nUse default Metaalbum\n\n")
        optimizer = get_pretrained_optimizer("mtlbm/full")
        # Explicitly reset path for pretrained optimizer
        optimizer.reset_path(
            config.qt.experiment_base_dir
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
            path=config.qt.experiment_base_dir,
            n=n_of_configs_to_create
        )
        if config.qt.use_custom_objective:
            print("\nUse custom objective\n")
            # Replace the default objective function with our custom one
            # TODO: check seeding - sampling should be deterministic
            # TODO: config needs to change for each trial > optimizer.ask()
            tuner.f = lambda trial, trial_info: quicktune_wrapper(optimizer.ask(), trial_info, config)
            tuner.run(fevals=config.max_evaluations, trial_info=trial_info)
        else:
            print("\nUse default objective\n")
            tuner.run(fevals=config.max_evaluations)
        
    else:
        print("\nUse QuickTuner\n")
        tuner = QuickTuner(
            optimizer=optimizer,
            f=lambda trial, trial_info: quicktune_wrapper(optimizer.ask(), trial_info, config),
            path=config.qt.experiment_base_dir,
        )
        tuner.run(fevals=config.max_evaluations, time_budget=None, trial_info=trial_info)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
