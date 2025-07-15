import logging
import os
import pickle
from pathlib import Path

import hydra
import yaml
from neps import run
from omegaconf import DictConfig, OmegaConf

from src.classification_2d.objective_function_2d import run_2d_pipeline
from src.classification_2d.preprocess_data_2d import (get_max_batch_size,
                                                      load_brain_tumor_dataset)
from src.classification_3d.objective_function_3d import run_3d_pipeline
from src.classification_3d.preprocess_data_3d import load_3d_dataset
from src.utils.common_utils import (neps_space_to_dict, set_seed,
                                    yaml_to_neps_pipeline_space)


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
    based on experimental_setting.data.dimensionality.

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
        dict: Dictionary containing optimization metrics
    """
    dimensionality = experimental_setting.data.dimensionality.lower()

    if dimensionality == "2d":
        return run_2d_pipeline(
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
            experimental_setting=experimental_setting,
            dataset_dict=dataset_dict,
            num_classes=num_classes,
            **hyperparameters,
        )
    elif dimensionality == "3d":
        return run_3d_pipeline(
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
        experimental_setting.max_evaluations = 10
        experimental_setting.data.k_folds = 2
        experimental_setting.pipeline_space = "configs/pipeline_spaces/pipeline_space_developer_mode.yaml"  # TODO @Diane: Update this
        experimental_setting.training.number_of_epochs = 2

    # Convert YAML pipeline space configuration into NePS-compatible format
    # NePS requires a specific dictionary structure for hyperparameter definitions
    pipeline_space = yaml_to_neps_pipeline_space(experimental_setting.pipeline_space)

    # Print experimental setting and pipeline space
    print("\nexperimental setting: ", experimental_setting, "\npipeline space: ", pipeline_space, "\n")

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

    # TODO: Each NePS run should also run for different train (incl. val) / test folds!!!

    if experimental_setting.data.preload_data:
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
        data_path = Path(experimental_setting.data.path)
        cache_file = (
            data_path
            / "cache"
            / f"{experimental_setting.data.dataset}_bs{get_max_batch_size(pipeline_space)}.pkl"
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

            dimensionality = experimental_setting.data.dimensionality.lower()
            if dimensionality == "2d":
                if experimental_setting.data.dataset == "brain_tumor":
                    dataset_dict = load_brain_tumor_dataset(
                        data_path=experimental_setting.data.path, seed=experimental_setting.seed
                    )
                else:
                    raise ValueError(f"Unsupported dataset: {experimental_setting.data.dataset}.")
                num_classes = dataset_dict["num_classes"]
            elif dimensionality == "3d":  # TODO: Add 3D dataset loading
                dataset_dict = load_3d_dataset(
                    experimental_setting.data.dataset, data_path=experimental_setting.data.path, seed=experimental_setting.seed
                )
                num_classes = dataset_dict["num_classes"]
            else:
                raise ValueError(
                    f"Unsupported dimensionality: {dimensionality}. Must be either '2d' or '3d'"
                )

        print(f"Dataset '{experimental_setting.data.dataset}' loaded with {num_classes} classes")
    else:
        print("\nData will be loaded on-demand during training\n")

    # Run NePS optimization
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
        root_directory=experimental_setting.neps_directory,
        max_evaluations_total=(
            1 if "baseline" in str(experimental_setting.pipeline_space) else experimental_setting.max_evaluations
        ),
        overwrite_working_directory=False,
        ignore_errors=True,
        # max_cost_total=10,  # e.g., if one config evaluation carries a cost of 2, we can evaluate 5 configs
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
