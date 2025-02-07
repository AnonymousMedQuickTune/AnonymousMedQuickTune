"""
Preprocesses and caches datasets for faster experiment initialization.
"""

import os
import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data import get_data_loaders
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

    # Convert YAML pipeline space configuration into NePS-compatible format
    pipeline_space = yaml_to_neps_pipeline_space(config.pipeline_space)

    # Create cache directory in the same location as the dataset
    data_path = Path(config.data.path)
    cache_dir = data_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache filename based on dataset and batch size
    cache_file = cache_dir / f"{dataset}_bs{pipeline_space['batch_size'].upper}.pkl"

    if cache_file.exists():
        print(f"Cache file already exists at {cache_file}")
        print("Delete it manually if you want to regenerate the cache.")
        return

    # Initialize data loaders
    print(f"Loading dataset '{config.data.dataset}'...")
    train_loader, val_loader, num_classes = get_data_loaders(
        config.data.dataset,
        config.data.num_workers,
        batch_size=pipeline_space["batch_size"].upper,
        split="train",
        data_path=config.data.path,
    )

    # Cache the data loaders and num_classes
    print(f"\nSaving cache to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(
            {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "num_classes": num_classes,
            },
            f,
        )

    print("\nPreprocessing completed! You can now run experiments faster.")
    print(f"Dataset '{config.data.dataset}' cached with {num_classes} classes")


if __name__ == "__main__":
    main()
