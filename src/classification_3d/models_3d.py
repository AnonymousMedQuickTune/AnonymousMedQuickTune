from torch import nn
from torchvision import models

import monai

try:
    # local import
    from utils.parameterized_models import (
        ParameterizedDenseNet, ParameterizedResNet, ParameterizedEfficientNetBN, ParameterizedSwinUNETR, ParameterizedViT
    )
    from utils.densenet import DenseModel
except ImportError:
    from src.classification_3d.utils.parameterized_models import (
        ParameterizedDenseNet, ParameterizedResNet, ParameterizedEfficientNetBN, ParameterizedSwinUNETR, ParameterizedViT
    )
    from src.classification_3d.utils.densenet import DenseModel

# NOTE: Currently not embedded in the code
# TODO @Diane: Test this function and finish implementation
def get_pretrained_3d_model(model_config, hyperparameters=None, developer_mode=False):
    """
    Create and initialize a pretrainedfixed (non-parametrized) 3D model for medical image classification.

    Args:
        model_config (dict): Model configuration containing:
            - type (str): Type of model. Available options:
                * 'resnet50': ResNet50 - Balanced performance and speed
                * 'resnet101': ResNet101 - Deeper version for higher accuracy
                * 'swin_unetr': SwinUNETR - Efficient hierarchical transformer with windowed attention
                * 'flexible_unet': FlexibleUNet - Flexible UNet architecture
            - task (str): Type of task ('classification', etc.)
            - num_classes (int): Number of output classes
        hyperparameters (dict, optional): Not used for fixed models, kept for compatibility
        developer_mode (bool): Not used for fixed models, kept for compatibility

    Returns:
        nn.Module: Initialized PyTorch model with pretrained weights loaded
        
    Note:
        * All models use pretrained weights and instance normalization instead of batch normalization 
          Pretrained weights are available from MedicalNet
        * Normalization strategy:
        - CNNs (ResNet18, ResNet34, ResNet50, FlexibleUNet): Instance Normalization (batch_size=1 compatible)
        - Transformers (SwinUNETR): Layer Normalization (batch_size agnostic)
        This ensures optimal performance for medical image classification with batch_size=1.
    """
    model_type = model_config["type"]
    num_classes = model_config["num_classes"]

    # models with available pretrained weights for 3D images
    # ------------------------------------------------------------
    if model_type == "resnet50":
        model = monai.networks.nets.ResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            layers=(3, 4, 6, 3),  # ResNet50
            norm="instance",
            pretrained=True  # TODO @Diane:See https://github.com/Tencent/MedicalNet
        )
    
    elif model_type == "resnet101":
        model = monai.networks.nets.ResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            layers=(3, 4, 23, 3),  # ResNet101
            norm="instance",
            pretrained=True  # TODO @Diane: See https://github.com/Tencent/MedicalNet
        )

    elif model_type == "swin_unetr":
        model = monai.networks.nets.SwinUNETR(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            norm_name="layer",
            pretrained=True  # Pretrained weights available from: https://arxiv.org/pdf/2307.16896
        )
    
    elif model_type == "flexible_unet":
        model = monai.networks.nets.FlexibleUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            norm_name="instance",
            pretrained=True  # MedicalNet pretrained weights available  TODO @Diane: check this out
        )

    else:
        raise ValueError("Unknown model type: " + model_type)

    return model

# TODO @Diane: Update search spaces!
def get_3d_model(model_config, hyperparameters, developer_mode, image_size=None):
    """
    Create and initialize a parametrized 3D model for medical 3D image classification.
    
    This function provides access to customizable 3D medical image classification models
    where hyperparameters can be optimized using Neural Pipeline Search (NePS) or MedQuickTune.
    
    The function supports three experiment modes: HPO experiments with search space hyperparameters (developer_mode=False, run_mode="NePS" or run_mode="QuickTune"), 
    baseline experiments with fixed hyperparameters (developer_mode=False, run_mode="Baseline"), and developer mode with reduced model sizes 
    for laptop/CPU testing (developer_mode=True).
    
    Args:
        model_config (dict): Model configuration containing:
            - type (str): Type of model. Available options:
                * 'densenet': Parametrized DenseNet (can simulate both 121 and 201)
                * 'resnet': Parametrized ResNet (can simulate 18, 50 and 101)
                * 'swin_unetr': Parametrized SwinUNETR - Efficient hierarchical transformer with windowed attention
                * 'efficientnet': Parametrized EfficientNet-B0 - Efficient architecture
                * 'vit': Parametrized Vision Transformer - Classic transformer for images
                * 'densenet_natalia': Custom DenseNet implementation by Natalia
            - task (str): Type of task ('classification', etc.)
            - num_classes (int): Number of output classes
        hyperparameters (dict): Hyperparameter configuration for NePS optimization. 
            Model-specific parameters:
            
            DenseNet:
            - densenet_type (str): Type of DenseNet ("densenet121", "densenet169", "densenet201")
            - init_features (int): Initial number of features (default: 64)
            - growth_rate (int): Growth rate (default: 32)
            - bn_size (int): Bottleneck size (default: 4)
            - dropout_rate (float): Dropout probability (default: 0.0)
            - act (str): Activation function (default: "relu")
            
            ResNet:
            - resnet_type (str): Type of ResNet ("resnet18", "resnet34", "resnet50")
            - conv1_t_size (int): Kernel size for initial convolution (default: 7)
            - conv1_t_stride (int): Stride for initial convolution (default: 1)
            - no_max_pool (bool): Whether to skip max pooling (default: False)
            - widen_factor (float): Widen factor for channels (default: 1.0)
            - act (str): Activation function (default: "relu")
            
            EfficientNet:
            - efficientnet_type (str): Model variant name ("efficientnet-b0", "efficientnet-b1", "efficientnet-b2")
            
            SwinUNETR:
            - patch_size (int): Patch size (default: 2)
            - feature_size (int): Feature size (default: 24)
            - depths (tuple): Number of layers in each stage (default: (2, 2, 2, 2))
            - num_heads (tuple): Number of attention heads per stage (default: (3, 6, 12, 24))
            - window_size (int): Local window size (default: 7)
            - mlp_ratio (float): MLP ratio (default: 4.0)
            - drop_rate (float): Dropout probability (default: 0.0)
            - attn_drop_rate (float): Attention dropout rate (default: 0.0)
            - dropout_path_rate (float): Drop path rate (default: 0.0)
            
            ViT:
            - patch_size (tuple): Patch size (default: (8, 8, 4))
            - hidden_size (int): Hidden size (default: 768)
            - mlp_dim (int): MLP dimension (default: 3072)
            - num_layers (int): Number of transformer layers (default: 12)
            - num_heads (int): Number of attention heads (default: 12)
            - pos_embed (str): Position embedding type (default: "learnable")
            - dropout_rate (float): Dropout probability (default: 0.0)
            - qkv_bias (bool): Whether to use bias in QKV projection (default: False)
        developer_mode (bool): Whether to use reduced model sizes for faster development/testing
        image_size (tuple, optional): Image size for ViT model (required for ViT)

    Returns:
        nn.Module: Initialized PyTorch model with customizable architecture
        
    Note:
        * Normalization strategy:
        - CNNs (DenseNet, ResNet, EfficientNet): Instance Normalization (batch_size=1 compatible)
        - Transformers (SwinUNETR, ViT): Layer Normalization (batch_size agnostic)
        This ensures optimal performance for medical image classification with batch_size=1.
    """
    model_type = model_config["type"]
    num_classes = model_config["num_classes"]

    if model_type == "densenet":
        model = ParameterizedDenseNet(hyperparameters, num_classes, developer_mode)

    elif model_type == "resnet":
        model = ParameterizedResNet(hyperparameters, num_classes, developer_mode)

    elif model_type == "efficientnet":  # No developer_mode needed for EfficientNet
        model = ParameterizedEfficientNetBN(hyperparameters, num_classes)

    elif model_type == "swin_unetr":
        model = ParameterizedSwinUNETR(hyperparameters, num_classes, developer_mode)

    elif model_type == "vit":
        model = ParameterizedViT(hyperparameters, num_classes, developer_mode, image_size)
    
    elif model_type == "densenet_natalia":
        model = DenseModel(hyperparameters, num_classes)

    else:
        raise ValueError("Unknown model type: " + model_type)

    return model
