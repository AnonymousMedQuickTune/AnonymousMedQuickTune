from torch import nn
from torchvision import models

def get_model(model_config):  # TODO: Use models from https://docs.monai.io/en/stable/networks.html
    """
    Create and initialize a model based on the model configuration.

    Args:
        model_config (dict): Model configuration containing:
            - type (str): Type of model ('resnet', 'efficientnet', 'vit',
                         'convnext', 'swin', 'densenet', 'efficientnetv2', 'densenet201')
            - task (str): Type of task ('classification', etc.)
            - num_classes (int): Number of output classes

    Returns:
        nn.Module: Initialized PyTorch model
    """
    model_type = model_config["type"]
    num_classes = model_config["num_classes"]

    # Modern, widely used architectures
    if model_type == "vit":  # Vision Transformer - State of the art
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Linear(model.hidden_dim, num_classes)
    elif model_type == "convnext":  # Modern CNN architecture
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_type == "resnet":  # Classic, reliable architecture
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "swin":  # Modern hierarchical ViT
        model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_type == "efficientnet":  # Efficient modern CNN
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "efficientnetv2":  # Updated EfficientNet
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "densenet":  # Older architecture
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_type == "densenet201":  # Larger DenseNet variant
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unknown model type: " + model_type)

    return model