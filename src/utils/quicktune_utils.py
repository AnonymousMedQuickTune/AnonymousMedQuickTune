import os
from pathlib import Path
import numpy as np
import pandas as pd
import traceback
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
                         EqualsCondition,
                         UniformFloatHyperparameter,
                         UniformIntegerHyperparameter)

logger = logging.getLogger(__name__)  # Add this line to create logger instance


def convert_numpy_to_python(value):
    """
    Convert numpy types to native Python types for OmegaConf compatibility.
    
    OmegaConf doesn't support numpy types (np.str_, np.int64, np.float64, np.bool_),
    so we need to convert them to native Python types.
    This function is compatible with both NumPy 1.x and 2.x.
    
    Args:
        value: Value that might be a numpy type
    
    Returns:
        Native Python type equivalent
    """
    # Check for numpy string types
    if isinstance(value, np.str_):
        return str(value)
    # Check for numpy integer types using base class (works with NumPy 1.x and 2.x)
    elif isinstance(value, np.integer):
        return int(value)
    # Check for numpy float types using base class (works with NumPy 1.x and 2.x)
    elif isinstance(value, np.floating):
        return float(value)
    # Check for numpy boolean types
    elif isinstance(value, np.bool_):
        return bool(value)
    # Check for numpy arrays
    elif isinstance(value, np.ndarray):
        return value.tolist()
    else:
        return value


def extract_and_pad_learning_curve(pipeline_dir: Path, expected_length: int, metric: str = "auc"):
    """
    Extract learning curve from training logs and pad it to expected length.
    
    This function extracts validation metrics from the metrics.csv log file and pads
    the curve to the expected length by repeating the last value. This is critical when
    early stopping is enabled, as actual training epochs may be less than expected.
    
    Args:
        pipeline_dir (Path): Directory containing training logs
        expected_length (int): Expected number of epochs (curve length)
        metric (str): Metric to extract (default: "auc")
    
    Returns:
        np.ndarray | None: Learning curve array of shape (1, expected_length) for QuickTune compatibility,
        or None if extraction fails. The 2D shape [1, n_epochs] matches QuickTune's expected format.
    """
    try:
        # Find metrics.csv file in the pipeline directory
        # Metrics are logged per fold, so we need to aggregate across folds
        metrics_files = list(pipeline_dir.glob("**/metrics.csv"))
        
        if not metrics_files:
            print(f"[Learning Curve] Warning: No metrics.csv files found in {pipeline_dir}")
            return None
        
        # Extract validation metrics from all folds and aggregate by epoch
        # We need to average metrics across folds for each epoch
        fold_curves = []
        max_epochs = 0
        
        for metrics_file in metrics_files:
            try:
                df = pd.read_csv(metrics_file)
                # Filter validation metrics
                val_df = df[df["phase"] == "val"].copy()
                if len(val_df) > 0:
                    # Get the metric column (convert to percentage if needed)
                    metric_values = val_df[metric].values
                    # If values are in [0, 1] range, convert to percentage
                    if len(metric_values) > 0 and metric_values.max() <= 1.0:
                        metric_values = metric_values * 100
                    fold_curves.append(metric_values)
                    max_epochs = max(max_epochs, len(metric_values))
            except Exception as e:
                print(f"[Learning Curve] Warning: Could not read {metrics_file}: {e}")
                continue
        
        if not fold_curves:
            print(f"[Learning Curve] Warning: No validation metrics found")
            return None
        
        # Pad all fold curves to the same length (max epochs across all folds)
        # Then average across folds for each epoch
        padded_curves = []
        for fold_curve in fold_curves:
            if len(fold_curve) < max_epochs:
                # Pad with last value
                last_value = fold_curve[-1] if len(fold_curve) > 0 else 0.0
                padding = np.full(max_epochs - len(fold_curve), last_value)
                padded_curve = np.concatenate([fold_curve, padding])
            else:
                padded_curve = fold_curve[:max_epochs]
            padded_curves.append(padded_curve)
        
        # Average across folds for each epoch
        curve = np.mean(padded_curves, axis=0)
        actual_length = len(curve)
        
        # Pad curve to expected length by repeating the last value
        if actual_length < expected_length:
            last_value = curve[-1] if len(curve) > 0 else 0.0
            padding = np.full(expected_length - actual_length, last_value)
            curve = np.concatenate([curve, padding])
            print(f"[Learning Curve] Padded curve from {actual_length} to {expected_length} epochs")
        elif actual_length > expected_length:
            # Trim if longer than expected (shouldn't happen, but handle it)
            curve = curve[:expected_length]
            print(f"[Learning Curve] Trimmed curve from {actual_length} to {expected_length} epochs")
        
        # QuickTune expects curves as 2D array with shape [1, n_epochs] for single curves
        # This matches the portfolio format where curves have shape [n_configs, n_epochs]
        curve = curve.astype(np.float32)
        curve = curve.reshape(1, -1)  # Reshape to [1, n_epochs]
        
        return curve
    
    except Exception as e:
        print(f"[Learning Curve] Error extracting learning curve: {e}")
        traceback.print_exc()
        return None


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
    
    # Extract dataset name from path
    dataset_name = path_root.name
    
    # Check if this is a MedMNIST3D dataset (these don't have a separate directory)
    medmnist3d_datasets = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 
                           'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
    is_medmnist3d = dataset_name.lower() in medmnist3d_datasets
    
    # For MedMNIST3D datasets, skip path validation and use hardcoded meta-features
    if not is_medmnist3d:
        if not path_root.exists():
            raise ValueError(f"The specified path does not exist: {path_root}")
        if not path_root.is_dir():
            raise ValueError(f"The specified path is not a directory: {path_root}")

    num_samples = 0
    num_classes = 0
    num_features = 224
    num_channels = 3

    # Check if dataset is pre-split (only for non-MedMNIST3D datasets)
    is_presplit = False
    if not is_medmnist3d:
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
                dataset = "lipo"
                num_classes = 2
                input_channels = 1  # Grayscale for medical images
                input_size_h = 256
                input_size_w = 256
                input_size_d = 32
                modality = "MRI"
                total_num_samples = 114
            elif dataset_name == "hcc":
                dataset = "hcc"
                num_classes = 2
                input_channels = 1  # Grayscale for medical images
                input_size_h = 256
                input_size_w = 256
                input_size_d = 32
                modality = "MRI"
                total_num_samples = 497
            elif dataset_name == "bflair":
                dataset = "bflair"
                num_classes = 2
                input_channels = 1  # Grayscale for medical images
                input_size_h = 256
                input_size_w = 256
                input_size_d = 32
                modality = "MRI"
                total_num_samples = 497
            elif dataset_name == "liver":
                dataset = "liver"
                num_classes = 2
                input_channels = 1  # Grayscale for medical images
                input_size_h = 256
                input_size_w = 256
                input_size_d = 32
                modality = "MRI"
                total_num_samples = 186
            elif dataset_name == "desmoid":
                dataset = "desmoid"
                num_classes = 2
                input_channels = 1  # Grayscale for medical images
                input_size_h = 256
                input_size_w = 256
                input_size_d = 32
                modality = "MRI"
                total_num_samples = 203
            elif dataset_name == "gist":
                dataset = "gist"
                num_classes = 2
                input_channels = 1  # Grayscale for medical images
                input_size_h = 256
                input_size_w = 256
                input_size_d = 96
                modality = "CT"
                total_num_samples = 245
            elif dataset_name == "crlm":
                dataset = "crlm"
                num_classes = 2
                input_channels = 1  # Grayscale for medical images
                input_size_h = 256
                input_size_w = 256
                input_size_d = 96
                modality = "CT"
                total_num_samples = 77
            elif dataset_name == "hecktor":  
                dataset = "hecktor"
                num_classes = 2
                input_channels = 1  # Grayscale for medical images
                input_size_h = 256
                input_size_w = 256
                input_size_d = 96
                modality = "CT"
                total_num_samples = 597
            elif dataset_name == "organmnist3d":
                dataset = "organmnist3d"
                num_classes = 11
                input_channels = 1  # Grayscale for MedMNIST3D
                input_size_h = 28
                input_size_w = 28
                input_size_d = 28
                modality = "CT"
                total_num_samples = 1743
            elif dataset_name == "nodulemnist3d":
                dataset = "nodulemnist3d"
                num_classes = 2
                input_channels = 1
                input_size_h = 28
                input_size_w = 28
                input_size_d = 28
                modality = "CT"
                total_num_samples = 1633
            elif dataset_name == "adrenalmnist3d":
                dataset = "adrenalmnist3d"
                num_classes = 2
                input_channels = 1
                input_size_h = 28
                input_size_w = 28
                input_size_d = 28
                modality = "CT"
                total_num_samples = 1584
            elif dataset_name == "fracturemnist3d":
                dataset = "fracturemnist3d"
                num_classes = 3
                input_channels = 1
                input_size_h = 28
                input_size_w = 28
                input_size_d = 28
                modality = "CT"
                total_num_samples = 1370
            elif dataset_name == "vesselmnist3d":
                dataset = "vesselmnist3d"
                num_classes = 2
                input_channels = 1
                input_size_h = 28
                input_size_w = 28
                input_size_d = 28
                modality = "MRA"
                total_num_samples = 1909
            elif dataset_name == "synapsemnist3d":
                dataset = "synapsemnist3d"
                num_classes = 2
                input_channels = 1
                input_size_h = 28
                input_size_w = 28
                input_size_d = 28
                modality = "Electron Microscope"
                total_num_samples = 1759
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

        except Exception as e:
            raise ValueError(f"Could not process dataset directory: {str(e)}")

    # Output as needed for QuickTune
    # TODO @Diane: for medical datasets might be good to add class distribution?: brain_tumor: {0: 98, 1: 155}
    metafeat = {
        "dataset": dataset,
        "num_classes": num_classes,
        "input_channels": input_channels,
        "input_size_h": input_size_h,
        "input_size_w": input_size_w,
        "input_size_d": input_size_d,
        "modality": modality,
        "total_num_samples": total_num_samples,
    }

    # For MedMNIST3D datasets, data-dir should point to the parent directory (data.path)
    # since the dataset files are stored directly in data.path, not in a subdirectory
    if is_medmnist3d:
        data_dir = str(path_root.parent)  # Use parent directory (e.g., datasets/)
    else:
        data_dir = str(path_root)  # Use the dataset directory (e.g., datasets/lipo)
    
    trial_info = {
        "data-dir": data_dir,
        "train-split": train_split,
        "val-split": val_split,
        "num-classes": num_classes,
    }

    return trial_info, metafeat


class CustomCostPredictor(CostPredictor):
    """Custom CostPredictor with modified default parameters and active flag support"""
    
    def __init__(self, path: str | None = None, seed: int | None = None, pipeline_space_path: str | None = None):
        # Initialize Predictor first with our name
        Predictor.__init__(self, name="medical_cost_predictor", path=path)
        
        # Store pipeline space path for active flag preprocessing during prediction
        self.pipeline_space_path = pipeline_space_path
        
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
    
    def _preprocess_predict_data(self, df: pd.DataFrame, fill_missing=True):
        """Override to add active flags before preprocessing"""
        from src.utils.portfolio_preprocessing import preprocess_portfolio_for_quicktune
        
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        # Add active flags if pipeline_space_path is available
        if self.pipeline_space_path is not None:
            df = preprocess_portfolio_for_quicktune(
                df=df,
                pipeline_space_path=self.pipeline_space_path,
                add_active_flags=True,
                handle_inactive_categorical=True,
                inactive_categorical_value="__inactive__"
            )
        
        # Call parent method
        return super()._preprocess_predict_data(df, fill_missing=fill_missing)


class CustomPerfPredictor(PerfPredictor):
    """Custom PerfPredictor with modified default parameters and active flag support"""
    
    def __init__(self, path: str | None = None, seed: int | None = None, pipeline_space_path: str | None = None):
        # Initialize Predictor first with our name
        Predictor.__init__(self, name="medical_perf_predictor", path=path)
        
        # Store pipeline space path for active flag preprocessing during prediction
        self.pipeline_space_path = pipeline_space_path
        
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
    
    def _preprocess_predict_data(self, df: pd.DataFrame, fill_missing=True):
        """Override to add active flags before preprocessing"""
        from src.utils.portfolio_preprocessing import preprocess_portfolio_for_quicktune
        
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        # Add active flags if pipeline_space_path is available
        if self.pipeline_space_path is not None:
            df = preprocess_portfolio_for_quicktune(
                df=df,
                pipeline_space_path=self.pipeline_space_path,
                add_active_flags=True,
                handle_inactive_categorical=True,
                inactive_categorical_value="__inactive__"
            )
        
        # Call parent method
        return super()._preprocess_predict_data(df, fill_missing=fill_missing)


class FTPFNPerfPredictor(PerfPredictor):
    """Performance predictor using FT-PFN (like in IfBO) instead of Gaussian Process regression."""

    def __init__(self, path: str | None = None, seed: int | None = None, pipeline_space_path: str | None = None):
        # Initialize Predictor first with our name
        Predictor.__init__(self, name="ft_pfn_medical_perf_predictor", path=path)
        
        # Store pipeline space path for active flag preprocessing during prediction
        self.pipeline_space_path = pipeline_space_path

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
    
    def _preprocess_predict_data(self, df: pd.DataFrame, fill_missing=True):
        """Override to add active flags before preprocessing"""
        from src.utils.portfolio_preprocessing import preprocess_portfolio_for_quicktune
        
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        # Add active flags if pipeline_space_path is available
        if self.pipeline_space_path is not None:
            df = preprocess_portfolio_for_quicktune(
                df=df,
                pipeline_space_path=self.pipeline_space_path,
                add_active_flags=True,
                handle_inactive_categorical=True,
                inactive_categorical_value="__inactive__"
            )
        
        # Call parent method
        return super()._preprocess_predict_data(df, fill_missing=fill_missing)

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

        # First pass: create all hyperparameters
        hyperparams = {}
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

            hyperparams[param_name] = param
            cs.add_hyperparameter(param)

        # Second pass: add conditions for conditional hyperparameters
        for param_name, param_config in config_dict.items():
            if "condition" in param_config:
                condition_config = param_config["condition"]
                parent_param_name = condition_config.get("parent")
                parent_value = condition_config.get("value")
                
                if parent_param_name not in hyperparams:
                    raise ValueError(
                        f"Condition parent parameter '{parent_param_name}' not found for '{param_name}'"
                    )
                
                parent_param = hyperparams[parent_param_name]
                child_param = hyperparams[param_name]
                
                condition = EqualsCondition(child_param, parent_param, parent_value)
                cs.add_condition(condition)

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
        
    
    
