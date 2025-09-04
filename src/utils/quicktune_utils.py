import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms  # type: ignore
from torchvision.datasets import ImageFolder  # type: ignore
from omegaconf import DictConfig
from ConfigSpace import ConfigurationSpace
from qtt.predictors import PerfPredictor, CostPredictor, Predictor
from qtt.optimizers.quick import QuickOptimizer
from qtt.optimizers.optimizer import Optimizer
from qtt.tuners.image.classification.tuner import QuickImageCLSTuner
from qtt.tuners.quick import QuickTuner 
from dataclasses import dataclass
import logging
from qtt.utils import set_logger_verbosity
from qtt.predictors.perf import DEFAULT_FIT_PARAMS as PERF_DEFAULT_FIT_PARAMS
from qtt.predictors.cost import DEFAULT_FIT_PARAMS as COST_DEFAULT_FIT_PARAMS
from src.utils.ftpfn import FTPFNSurrogateModel  # , FTPFN, FTPFNSurrogateModel2

from ConfigSpace import (CategoricalHyperparameter, ConfigurationSpace,
                         UniformFloatHyperparameter,
                         UniformIntegerHyperparameter)

logger = logging.getLogger(__name__)  # Add this line to create logger instance

def custom_extract_image_dataset_metafeat(
    path_root: str | Path, train_split: str = "train", val_split: str = "val"
):
    """
    Extracts metadata features from an image dataset for classification tasks.
    Automatically detects if the dataset is pre-split (train/val) or unsplit.

    This function analyzes the specified dataset directory to compute metadata
    features, such as the number of samples, number of classes, average number
    of features (image size), and number of channels.

    Args:
        path_root (str | Path): The root directory of the dataset.
        train_split (str, optional): The subdirectory name for training data. Defaults to "train".
        val_split (str, optional): The subdirectory name for validation data. Defaults to "val".

    Returns:
        tuple: A tuple containing:
            - trial_info (dict): Information about the dataset directory and splits.
            - metafeat (dict): Metadata features including:
                - "num-samples": Total number of samples in the dataset.
                - "num-classes": Number of classes in the dataset.
                - "num-features": Average number of features (image size).
                - "num-channels": Number of channels in the images.

    Raises:
        ValueError: If the specified path does not exist or is not a directory.
    """
    # handle path
    path_root = Path(path_root)
    path_root = path_root.expanduser()  # expands ~ to home directory
    path_root = path_root.resolve()  # convert to an absolute path
    if not path_root.exists():
        raise ValueError(f"The specified path does not exist: {path_root}")
    if not path_root.is_dir():
        raise ValueError(f"The specified path is not a directory: {path_root}")

    num_samples = 0
    num_classes = 0
    num_features = 224
    num_channels = 3

    # Check if dataset is pre-split
    train_path = path_root / train_split
    val_path = path_root / val_split
    is_presplit = train_path.exists() or val_path.exists()

    if is_presplit:
        # QuickTune's default extract_image_dataset_metafeat() implementation
        # Handle pre-split dataset (existing logic)
        if train_path.exists():
            trainset = ImageFolder(train_path)
            num_samples += len(trainset)
            num_channels = 3 if trainset[0][0].mode == "RGB" else 1
            num_classes = len(trainset.classes)

            for img, _ in trainset:
                num_features += img.size[0]
            num_features //= len(trainset)

        if os.path.exists(val_path):
            valset = ImageFolder(val_path)
            num_samples += len(valset)
    
    else:
        # Handle unsplit dataset (single directory)
        try:
            # Dataset-specific meta-features
            dataset_name = path_root.name
            if dataset_name == "brain_tumor":  # calculated in load_brain_tumor_dataset()
                num_samples = 253
                num_classes = 2
                num_features = 224
                num_channels = 3
            elif dataset_name == "lipo":
                num_samples = 114  # Actual values from dataset statistics
                num_classes = 2
                num_features = 224  # Standard input size for medical images  # TODO @Diane: Update this
                num_channels = 1  # Grayscale for medical images
            elif dataset_name == "desmoid":
                num_samples = 203  # Actual values from dataset statistics
                num_classes = 2
                num_features = 224  # Standard input size for medical images  # TODO @Diane: Update this
                num_channels = 1  # Grayscale for medical images
            else:
                raise ValueError(f"Unsupported dataset: {experimental_setting.data.dataset}")

        except Exception as e:
            raise ValueError(f"Could not process dataset directory: {str(e)}")

    # Output as needed for QuickTune
    # TODO @Diane: for medical datasets might be good to add class distribution?: brain_tumor: {0: 98, 1: 155}
    metafeat = {
        "num-samples": num_samples,
        "num-classes": num_classes,
        "num-features": num_features,
        "num-channels": num_channels,
    }

    trial_info = {
        "data-dir": str(path_root),
        "train-split": train_split,
        "val-split": val_split,
        "num-classes": num_classes,
    }

    return trial_info, metafeat


class CustomCostPredictor(CostPredictor):
    """Custom CostPredictor with modified default parameters"""
    
    def __init__(self, path: str | None = None, seed: int | None = None):
        # Initialize Predictor first with our name
        Predictor.__init__(self, name="medical_cost_predictor", path=path)
        
        # Override default parameters
        # Batch size needs to be reduced to avoid division by zero for small datasets
        custom_defaults = {
            "learning_rate_init": 0.0001,
            "batch_size": 1,  # default: 1024
            "max_iter": 100,
            "early_stop": True,
            "patience": 5,
            "validation_fraction": 0.1,
            "tol": 1e-4,
        }
        verbosity = 2
        
        # Initialize CostPredictor without calling Predictor.__init__ again
        self.fit_params = self._validate_fit_params(custom_defaults, COST_DEFAULT_FIT_PARAMS)
        self.seed = seed
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)


class CustomPerfPredictor(PerfPredictor):
    """Custom PerfPredictor with modified default parameters"""
    
    def __init__(self, path: str | None = None, seed: int | None = None):
        # Initialize Predictor first with our name
        Predictor.__init__(self, name="medical_perf_predictor", path=path)
        
        # Override default parameters
        custom_fit_params = {
            "learning_rate_init": 0.0001,
            "batch_size": 1,  # default: 1024
            "max_iter": 100,
            "early_stop": True,
            "patience": 5,
            "validation_fraction": 0.1,
            "tol": 1e-4,
        }
        custom_refit_params = {
            "learning_rate_init": 0.001,
            "batch_size": 2048,  # TODO: check if this works
            "max_iter": 50,
            "early_stop": True,
            "patience": 5,
            "tol": 1e-4,
        }
        verbosity = 2
        
        # Initialize performance predictor attributes
        self.fit_params = self._validate_fit_params(custom_fit_params, PERF_DEFAULT_FIT_PARAMS)
        self.refit_params = self._validate_fit_params(custom_refit_params, PERF_DEFAULT_FIT_PARAMS)
        self.seed = seed
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)


class FTPFNPerfPredictor(PerfPredictor):
    """Performance predictor using FT-PFN (like in IfBO) instead of Gaussian Process regression."""

    def __init__(self, path: str | None = None, seed: int | None = None):
        # Initialize Predictor first with our name
        Predictor.__init__(self, name="ft_pfn_medical_perf_predictor", path=path)

        # Override default parameters
        custom_fit_params = {
            "learning_rate_init": 0.0001,
            "batch_size": 1,  # default: 1024
            "max_iter": 100,
            "early_stop": True,
            "patience": 5,
            "validation_fraction": 0.1,
            "tol": 1e-4,
        }
        custom_refit_params = {
            "learning_rate_init": 0.001,
            "batch_size": 2048,  # TODO: check if this works
            "max_iter": 50,
            "early_stop": True,
            "patience": 5,
            "tol": 1e-4,
        }
        verbosity = 2

        # Initialize performance predictor attributes
        self.fit_params = self._validate_fit_params(custom_fit_params, PERF_DEFAULT_FIT_PARAMS)
        self.refit_params = self._validate_fit_params(custom_refit_params, PERF_DEFAULT_FIT_PARAMS)
        self.seed = seed
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)

    def _get_model(self):
        """Override _get_model to return FTPFNSurrogateModel instead of default GP model"""    
        params = {
            "in_dim": [
                len(self.types_of_features["continuous"]),
                len(self.types_of_features["categorical"]) + len(self.types_of_features["bool"]),
            ],
            "in_curve_dim": self._curve_dim,
        }
        return FTPFNSurrogateModel(**params)
        # ------------------------------------------------------------------------------------------
        print("Parameters being passed to FTPFN:")
        print(params)
        
        # FTPFN expects different parameters than FTPFNSurrogateModel
        ftpfn_params = {
            "target_path": None,  # or specify a path
            "version": "0.0.1",
            "device": None  # will default to CPU or available GPU
        }
        print("\nActual FTPFN parameters:")
        print(ftpfn_params)
        
        return FTPFN(**ftpfn_params)  # Use the correct parameters for FTPFN  # TODO @Diane: Double check this implementation
        # return FTPFNSurrogateModel(**params)
    

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
                pipeline_df=pd.read_csv(f"{portfolio_dir}/config.csv", index_col=0),
                curve_df=pd.read_csv(f"{portfolio_dir}/curve.csv", index_col=0),
                cost_df=pd.read_csv(f"{portfolio_dir}/cost.csv", index_col=0),
                meta_df=pd.read_csv(f"{portfolio_dir}/meta.csv", index_col=0),
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Portfolio file not found in {portfolio_dir}: {e}")

    @staticmethod
    def save(portfolio: PortfolioData, output_dir: str):
        """Save portfolio data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        portfolio.pipeline_df.to_csv(f"{output_dir}/config.csv", index=True)
        portfolio.curve_df.to_csv(f"{output_dir}/curve.csv", index=True)
        portfolio.cost_df.to_csv(f"{output_dir}/cost.csv", index=True)
        portfolio.meta_df.to_csv(f"{output_dir}/meta.csv", index=True)

def save_config_files(output_dir: Path, config_files: list) -> None:
    """Save configuration files to the specified directory.
    
    Args:
        output_dir (Path): Directory to save the configuration files
        config_files (list): List of tuples containing (filename, data)
    """
    for filename, data in config_files:
        config_path = output_dir / filename
        try:
            config_path.write_text(data, encoding="utf-8")
        except IOError as e:
            logging.error(f"Failed to write configuration file {filename}: {e}")
            raise RuntimeError(f"Configuration saving failed: {e}") from e
    
class CustomQuickImageCLSTuner(QuickImageCLSTuner):
    """Custom QuickImageCLSTuner that uses our optimizer instead of loading MetaAlbum"""
    
    def __init__(
        self,
        data_path: str,
        optimizer: QuickOptimizer,
        n: int = 512,
        path: str | None = None,
        verbosity: int = 2,
    ):
        # Override the default initialization to prevent MetaAlbum loading
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)
        
        # Use our custom metafeature extraction
        trial_info, metafeat = custom_extract_image_dataset_metafeat(data_path)
        self.trial_info = trial_info
        
        # Setup optimizer with our metafeatures
        optimizer.setup(n, metafeat=metafeat)
        
        # Initialize parent class but skip its optimizer initialization
        super(QuickImageCLSTuner, self).__init__(  # Note: Call QuickTuner's init
            optimizer=optimizer,
            f=None,
            path=path,
            verbosity=verbosity
        )
        
    
    