#!/usr/bin/env python3
"""
NePS to QuickTune Output Adapter

This script converts NePS optimization results into the format required by QuickTune.
It creates three CSV files:
- config.csv: Hyperparameter configurations
- curve.csv: Learning curves
- cost.csv: Runtime costs

Author: [Your Name]
Date: [Current Date]
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


class NePSQuickTuneAdapter:
    def __init__(self, input_path: str, output_dir: str):
        """
        Initialize the adapter with input and output paths.

        Args:
            input_path: Path to the NePS output file
            output_dir: Directory where the CSV files should be saved
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.setup_logging()

    @staticmethod
    def setup_logging() -> None:
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def parse_neps_output(self) -> List[Dict[str, Any]]:
        """
        Parse the NePS output file into a list of dictionaries.

        Returns:
            List of dictionaries containing parsed results
        """
        results = []
        try:
            with open(self.input_path, "r") as file:
                current_entry = {}
                for line in file:
                    line = line.strip()

                    if line.startswith("Loss:"):
                        current_entry["loss"] = float(line.split(":")[1].strip())
                    elif line.startswith("Config ID:"):
                        current_entry["config_id"] = line.split(":")[1].strip()
                    elif line.startswith("Config:"):
                        config_str = line[line.index("{") :].strip()
                        try:
                            config_dict = eval(
                                config_str
                            )  # Safe since we know the format
                            current_entry.update(config_dict)
                            results.append(current_entry.copy())
                            current_entry = {}
                        except (SyntaxError, ValueError) as e:
                            logging.error(f"Failed to parse config: {config_str}")
                            logging.error(f"Error: {str(e)}")

        except FileNotFoundError:
            logging.error(f"Input file not found: {self.input_path}")
            raise

        logging.info(f"Successfully parsed {len(results)} configurations")
        return results

    def create_dataframes(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create DataFrames for configurations, learning curves, and costs.

        Args:
            results: List of parsed results

        Returns:
            Tuple of (config_df, curves_df, cost_df)
        """
        # Create configurations DataFrame
        config_data = {
            "batch_size": [r["batch_size"] for r in results],
            "learning_rate": [r["learning_rate"] for r in results],
            "number_of_epochs": [r["number_of_epochs"] for r in results],
        }
        config_df = pd.DataFrame(config_data)

        # Create learning curves DataFrame
        curves_data = {}
        for i, result in enumerate(results):
            # Convert loss to accuracy-like metric (0-1 range)
            acc = -result["loss"] / 100.0
            curves_data[i] = [acc]
        curves_df = pd.DataFrame(curves_data).T

        # Create cost DataFrame
        cost_data = {"cost": [r["number_of_epochs"] for r in results]}
        cost_df = pd.DataFrame(cost_data)

        return config_df, curves_df, cost_df

    def save_dataframes(
        self, config_df: pd.DataFrame, curves_df: pd.DataFrame, cost_df: pd.DataFrame
    ) -> None:
        """
        Save DataFrames to CSV files.

        Args:
            config_df: Configuration DataFrame
            curves_df: Learning curves DataFrame
            cost_df: Cost DataFrame
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            config_df.to_csv(self.output_dir / "config.csv")
            curves_df.to_csv(self.output_dir / "curve.csv")
            cost_df.to_csv(self.output_dir / "cost.csv")
            logging.info(f"Successfully saved CSV files to {self.output_dir}")
        except Exception as e:
            logging.error(f"Failed to save CSV files: {str(e)}")
            raise

    def convert(self) -> None:
        """Execute the complete conversion process."""
        logging.info(f"Starting conversion from {self.input_path}")
        results = self.parse_neps_output()
        config_df, curves_df, cost_df = self.create_dataframes(results)
        self.save_dataframes(config_df, curves_df, cost_df)
        logging.info("Conversion completed successfully")


def main():
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(
        description="Convert NePS output to QuickTune input format"
    )
    parser.add_argument("input_path", help="Path to the NePS output file")
    parser.add_argument(
        "--output-dir",
        default="quicktune_input",
        help="Directory to save the output CSV files (default: quicktune_input)",
    )

    args = parser.parse_args()

    try:
        adapter = NePSQuickTuneAdapter(args.input_path, args.output_dir)
        adapter.convert()
    except Exception as e:
        logging.error(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
