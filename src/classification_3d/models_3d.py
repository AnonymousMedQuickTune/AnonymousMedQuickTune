from torch import nn
from torchvision import models


def get_3d_model(
    model_config,
):  # TODO: Use models from https://docs.monai.io/en/stable/networks.html
    """
    Create and initialize a model based on the model configuration.

    Args:
        model_config (dict): Model configuration containing:
            - type (str): Type of model ('resnet', TODO: Add other models)
            - task (str): Type of task ('classification', etc.)
            - num_classes (int): Number of output classes

    Returns:
        nn.Module: Initialized PyTorch model
    """
    model_type = model_config["type"]
    num_classes = model_config["num_classes"]

    # Modern, widely used architectures
    if model_type == "resnet":
        model = None  # TODO: Add 3D ResNet
    else:
        raise ValueError("Unknown model type: " + model_type)

    return model
