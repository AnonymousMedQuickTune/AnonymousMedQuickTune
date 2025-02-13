"""
Preprocesses and caches datasets for faster experiment initialization.
"""

import os
import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data import (calculate_normalization_stats, get_data_loaders,
                      load_dataset)
from src.util_functions import yaml_to_neps_pipeline_space


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main_experiment_config.yaml",
)
def main(config: DictConfig) -> None:
    """
    Preprocess and cache datasets for faster experiment initialization.

    Args:
        config (DictConfig): Hydra configuration object
    """
    print("\nPreprocessing datasets...")

    # Get dataset name from config
    dataset = config.data.dataset
    print(f"Processing dataset: {dataset}")

    # Create cache directory in the same location as the dataset
    data_path = Path(config.data.path)
    cache_dir = data_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache filename - simplified to just use dataset name
    cache_file = cache_dir / f"{dataset}_normalization_stats.pkl"

    if cache_file.exists():
        print(f"Cache file already exists at {cache_file}")
        print("Delete it manually if you want to regenerate the cache.")
        return

    # Load raw dataset first
    print(f"Loading dataset '{dataset}'...")
    dataset_dict = load_dataset(dataset, data_path=config.data.path)

    # Calculate normalization statistics from training data only
    print("Calculating dataset-specific normalization statistics...")
    means, stds = calculate_normalization_stats(dataset_dict["train_data"])
    print(f"Dataset means: {means}")
    print(f"Dataset stds: {stds}")

    # Cache the normalization stats
    print(f"\nSaving cache to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(
            {"normalization_stats": (means, stds)},
            f,
        )

    print("\nPreprocessing completed!")
    print(f"Dataset '{dataset}' preprocessed")
    print(f"Dataset-specific normalization values have been calculated and cached.")


if __name__ == "__main__":
    main()
