from torch import nn
from torchvision import models

import monai

try:
    # local import
    from utils.densenet import DenseModel
except ImportError:
    from src.classification_3d.utils.densenet import DenseModel


def get_3d_model(
    model_config,
    hyperparameters,
):  # TODO: Use models from https://docs.monai.io/en/stable/networks.html  -> see densenet.py
    """
    Create and initialize a model based on the model configuration.

    Args:
        model_config (dict): Model configuration containing:
            - type (str): Type of model ('densenetv1', 'densenetv2')
            - task (str): Type of task ('classification', etc.)
            - num_classes (int): Number of output classes

    Returns:
        nn.Module: Initialized PyTorch model
    """
    model_type = model_config["type"]
    num_classes = model_config["num_classes"]

    # Modern, widely used architectures
    if model_type == "densenetv1":
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes)

    elif model_type == "densenetv2":
        model = DenseModel(hyperparameters)

    else:
        raise ValueError("Unknown model type: " + model_type)

    return model
