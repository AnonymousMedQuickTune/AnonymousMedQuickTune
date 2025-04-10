import hydra
from omegaconf import DictConfig, OmegaConf
import os
from dataclasses import dataclass
from pathlib import Path
import logging
import time

import pandas as pd
import yaml
from ConfigSpace import (CategoricalHyperparameter, ConfigurationSpace,
                        UniformFloatHyperparameter, UniformIntegerHyperparameter)
from qtt import QuickOptimizer, QuickTuner, get_pretrained_optimizer
from qtt.finetune.image.classification.utils import extract_image_dataset_metafeat

from src.classification_2d.objective_function_2d import run_2d_pipeline
from src.classification_2d.preprocess_data_2d import load_brain_tumor_dataset
from src.utils.common_utils import set_seed
import traceback

# Constants
CONFIG_PATH = Path("configs/pipeline_spaces/pipeline_space_without_user_priors.yaml")
PORTFOLIO_FILES = ["config.csv", "curve.csv", "cost.csv", "meta.csv"]

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
            'float': UniformFloatHyperparameter,
            'int': UniformIntegerHyperparameter,
            'categorical': CategoricalHyperparameter
        }
        
        for param_name, param_config in config_dict.items():
            param_type = param_config['type']
            param_class = type_to_param.get(param_type)
            
            if not param_class:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            # Convert scientific notation strings to float
            if param_type in ['float', 'int']:
                lower = float(param_config['lower']) if isinstance(param_config['lower'], str) else param_config['lower']
                upper = float(param_config['upper']) if isinstance(param_config['upper'], str) else param_config['upper']
                
                if param_type == 'float':
                    param = param_class(
                        name=param_name,
                        lower=lower,
                        upper=upper,
                        log=param_config.get('log', False)
                    )
                else:  # int
                    param = param_class(
                        name=param_name,
                        lower=int(lower),
                        upper=int(upper)
                    )
            else:  # categorical
                param = param_class(name=param_name, choices=param_config['choices'])
                
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
                meta_df=pd.read_csv(f"{portfolio_dir}/meta.csv")
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
    
    Args:
        trial (dict): Trial configuration from QuickTune
        trial_info (dict): Trial information from QuickTune
        config (DictConfig): Hydra configuration object
    """
    start_time = time.time()
    
    # Prepare directories
    pipeline_dir = os.path.join(trial_info["output-dir"], str(trial["config-id"]))
    prev_pipeline_dir = None
    
    # Load dataset
    data_dir = os.path.dirname(trial_info["data-dir"])
    dataset_dict = load_brain_tumor_dataset(data_dir)
    num_classes = 2
    
    try:
        # Merge trial config with fidelity as number_of_epochs
        trial_config = trial["config"].copy()
        # trial_config["number_of_epochs"] = 20  # trial["fidelity"]
        print(f"\n\n\n\nTrial config: {trial_config}\n\n\n\n")
        
        result = run_2d_pipeline(
            pipeline_directory=pipeline_dir,
            previous_pipeline_directory=prev_pipeline_dir,
            config=config,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            **trial_config
        )
        
        # Extract metrics safely
        info_dict = result.get("extra", {})
        final_metrics = info_dict.get("all_folds_final_metrics", {})
        
        # Get the metric specified in config or default to 0.0
        score = final_metrics.get(config.metric, 0.0)
        if score is None:
            score = 0.0
            
        # Convert score to percentage if needed
        if isinstance(score, (int, float)) and score <= 1.0:
            score = score * 100
            
        return {
            "config-id": trial["config-id"],
            "status": "SUCCESS",
            "score": score,
            "cost": time.time() - start_time,
            "fidelity": trial_config["number_of_epochs"],
            "config": trial_config
        }
    except Exception as e:
        print(f"Error in pipeline: {e}")
        traceback.print_exc()  # Now traceback is imported
        return {
            "config-id": trial["config-id"],
            "status": "FAILED",
            "score": float("-inf"),
            "cost": float("inf"),
            "fidelity": trial["fidelity"],
            "config": trial["config"]
        }


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main_experiment_config.yaml",
)
def main(config: DictConfig) -> None:
    """
    Main entry point for QuickTune optimization.
    
    Args:
        config (DictConfig): Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(config.seed)

    # Create quicktune output directory with full path structure
    experiment_path = os.path.join(
        "experiments",
        "quicktune",
        config.data.dataset,
        config.experiment_name,
        f"seed_{config.seed}"
    )
    os.makedirs(experiment_path, exist_ok=True)
    abs_experiment_path = os.path.abspath(experiment_path)

    # Load original pipeline space configuration
    with open(config.pipeline_space, "r", encoding="utf-8") as f:
        pipeline_space = yaml.safe_load(f)

    # Create ConfigurationSpace directly from the original YAML
    configspace = ConfigSpaceBuilder.from_yaml(pipeline_space)

    # Print main experiment configuration and pipeline space
    print("\nconfig: ", config, "\npipeline space: ", pipeline_space, "\n")

    # Create directory for configuration files and logs
    output_dir = os.path.join(config.experiment_base_dir, "hydra_output")
    os.makedirs(output_dir, exist_ok=True)

    # Save configurations
    config_files = [
        ("config.yaml", OmegaConf.to_yaml(config)),
        ("pipeline_space.yaml", yaml.dump(pipeline_space, default_flow_style=False)),
    ]

    for filename, data in config_files:
        config_path = os.path.join(output_dir, filename)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(data)
        except IOError as e:
            logging.error(f"Failed to write configuration file {filename}: {e}")

    # if config.searcher == "quicktune_medical":
    if True:
        # Load portfolio data
        portfolio = PortfolioManager.load(config.portfolio_dir)
    
        # Initialize optimizer with portfolio
        merged_df = pd.merge(portfolio.pipeline_df, portfolio.meta_df, on="dataset")
        merged_df = merged_df.drop(columns=["dataset"])

        optimizer = QuickOptimizer(
            cs=configspace,
            max_fidelity=50,  # Using the max_budget value as max_fidelity
            cost_aware=True,
            path=abs_experiment_path  # Set the path during initialization
        )
        optimizer.reset_path(abs_experiment_path)  # Explicitly reset path to ensure it propagates to predictors
    else:
        # Initialize the optimizer with pretrained model
        optimizer = get_pretrained_optimizer("mtlbm/full")
        optimizer.reset_path(abs_experiment_path)  # Explicitly reset path for pretrained optimizer

    print("\nOptimizer created\n")

    # Extract meta-features from the dataset directory
    if config.data.dataset == "brain_tumor":
        trial_info, metafeat = extract_image_dataset_metafeat(
            path_root=os.path.join(config.data.path, "brain_tumor"),
            train_split="train",
            val_split="val"
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")
    
    print("\nMeta-features extracted\n")
    
    # Setup optimizer with target dataset
    optimizer.setup(n=128, metafeat=metafeat)
    print("\nOptimizer setup complete\n")
    
    # Create QuickTuner instance with experiment-specific output path
    qt = QuickTuner(
        optimizer=optimizer, 
        f=lambda trial, trial_info: quicktune_wrapper(trial, trial_info, config),
        path=experiment_path  # Use the experiment-specific path
    )
    qt.run(
        fevals=config.max_evaluations, 
        time_budget=None, 
        trial_info=trial_info
    )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
