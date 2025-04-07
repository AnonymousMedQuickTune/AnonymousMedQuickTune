import logging
import os
import pickle
from pathlib import Path

import hydra
import yaml
from neps import run
from omegaconf import DictConfig, OmegaConf

from src.classification_2d.objective_function_2d import run_2d_pipeline
from src.classification_2d.preprocess_brain_tumor_data_2d import (load_brain_tumor_dataset,
                                                                  get_max_batch_size)
from src.classification_3d.objective_function_3d import run_3d_pipeline
from src.classification_3d.preprocess_data_3d import load_3d_dataset
from src.utils.common_utils import set_seed, yaml_to_neps_pipeline_space


def run_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    config,
    dataset_dict,
    num_classes,
    **hyperparameters,
):
    """
    Main pipeline function that delegates to specific 2D or 3D implementations
    based on config.data.dimensionality.

    NOTE: The argument order and parameter names must strictly follow NePS conventions
    to ensure proper optimization and checkpointing functionality.

    Args:
        pipeline_directory (str): Directory where current pipeline results will be saved
        previous_pipeline_directory (str): Directory containing previous pipeline runs
        config (DictConfig): Hydra configuration object
        dataset_dict (dict, optional): Combined train+val data and labels dictionary if preloaded
        num_classes (int, optional): Number of classes in the dataset if preloaded
        **hyperparameters: Configuration dictionary containing hyperparameters

    Returns:
        dict: Dictionary containing optimization metrics
    """
    dimensionality = config.data.dimensionality.lower()

    if dimensionality == "2d":
        return run_2d_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            config=config,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            **hyperparameters,
        )
    elif dimensionality == "3d":
        return run_3d_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            config=config,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            **hyperparameters,
        )
    else:
        raise ValueError(
            f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
        )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main_experiment_config.yaml",
)
def main(config: DictConfig) -> None:
    """
    Main entry point for the training script.

    Args:
        config (DictConfig): Hydra configuration object
    """
    # Set seed for NePS reproducibility
    set_seed(config.seed)

    # Convert YAML pipeline space configuration into NePS-compatible format
    # NePS requires a specific dictionary structure for hyperparameter definitions
    pipeline_space = yaml_to_neps_pipeline_space(config.pipeline_space)

    # Print main experiment configuration and pipeline space
    print("\nconfig: ", config, "\npipeline space: ", pipeline_space, "\n")

    # Create directory for configuration files and logs
    output_dir = os.path.join(config.experiment_base_dir, "hydra_output")
    os.makedirs(output_dir, exist_ok=True)

    # Load original pipeline space configuration for human-readable logging
    # This version maintains the original YAML structure without NePS-specific transformations
    with open(config.pipeline_space, "r", encoding="utf-8") as f:
        original_pipeline_space = yaml.safe_load(f)

    # Save different versions of configurations:
    # 1. Full Hydra config (includes all settings)
    # 2. NePS-compatible pipeline space (used for optimization)
    # 3. Original compact pipeline space (for better readability)
    config_files = [
        ("config.yaml", OmegaConf.to_yaml(config)),
        ("pipeline_space.yaml", yaml.dump(pipeline_space, default_flow_style=False)),
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

    # TODO: Each NePS run should also run for different train (incl. val) / test folds!!!

    if config.data.preload_data:
        # Preloading data has several benefits:
        # 1. Reduces I/O overhead during training iterations:
        #    - Loads data into memory once instead of repeatedly from disk
        #    - Minimizes disk access during training loops
        #    - Significantly improves training speed, especially with fast storage
        #    - Reduces system resource usage from repeated file operations
        # 2. Ensures consistent data loading across optimization runs:
        #    - Same data ordering and batching between different trials
        #    - Eliminates random variations from data loading
        #    - Improves reproducibility of experiments
        #    - Makes hyperparameter comparisons more reliable
        # 3. Enables early validation of data integrity and format:
        #    - Verify all data samples are properly loaded
        #    - Check for corrupted or missing data
        #    - Confirm data dimensions and types match expectations
        #    - Detect potential issues before starting expensive training runs
        # Note on Data Augmentation:
        # - Raw data is preloaded, but dynamic augmentation still occurs during training
        # - Each epoch can apply different random augmentations to the base data on-the-fly
        # - Memory efficiency is maintained as augmented versions aren't stored

        # Try to load from cache first
        data_path = Path(config.data.path)
        cache_file = (
            data_path
            / "cache"
            / f"{config.data.dataset}_bs{get_max_batch_size(pipeline_space)}.pkl"
        )

        # Cache mechanism further improves performance by avoiding repeated data processing
        if cache_file.exists():
            print("\nLoading data from cache...")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                # Directly use cached_data as it should already be in the correct format
                dataset_dict = cached_data
                num_classes = dataset_dict["num_classes"]
        else:
            print("\nNo cache found. Loading data directly...")

            dimensionality = config.data.dimensionality.lower()
            if dimensionality == "2d":
                if config.data.dataset == "brain_tumor":
                    dataset_dict = load_brain_tumor_dataset(
                        data_path=config.data.path, seed=config.seed
                    )
                else:
                    raise ValueError(f"Unsupported dataset: {config.data.dataset}.")
                num_classes = dataset_dict["num_classes"]
            elif dimensionality == "3d":  # TODO: Add 3D dataset loading
                dataset_dict = load_3d_dataset(
                    config.data.dataset, data_path=config.data.path, seed=config.seed
                )
                num_classes = dataset_dict["num_classes"]
            else:
                raise ValueError(
                    f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
                )

        print(f"Dataset '{config.data.dataset}' loaded with {num_classes} classes")
    else:
        print("\nData will be loaded on-demand during training\n")

    # Run NePS optimization
    logging.basicConfig(level=logging.INFO)
    run(
        run_pipeline=lambda pipeline_directory, previous_pipeline_directory, **kwargs: run_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            config=config,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            **kwargs,
        ),
        pipeline_space=pipeline_space,  # Hyperparameter search space
        searcher=config.searcher,  # HPO algorithm
        root_directory=config.root_directory,
        max_evaluations_total=(
            1 if "baseline" in str(config.pipeline_space) else config.max_evaluations
        ),
        overwrite_working_directory=False,
        # max_cost_total=10,  # e.g., if one config evaluation carries a cost of 2, we can evaluate 5 configs
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
