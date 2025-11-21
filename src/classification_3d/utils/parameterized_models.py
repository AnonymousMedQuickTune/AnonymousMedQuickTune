# General
import numpy as np
import pandas as pd
from collections import OrderedDict
from collections.abc import Sequence

# Torch
import torch
import torch.nn as nn

# MONAI
import monai
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.densenet import _DenseBlock, _Transition

# https://docs.monai.io/en/1.3.0/_modules/monai/networks/nets/densenet.html
# https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/densenet.py#L152-L256
class ParameterizedDenseNet(nn.Module):
    """
    Parametrized DenseNet for 3D medical image classification.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    
    Note:
        The architecture is NOT compatible with the Med3D pretrained weights: https://github.com/Tencent/MedicalNet
    """
    
    def __init__(self, hyperparameters: dict, num_classes: int, developer_mode: bool, is_medmnist: bool, run_mode: str):
        super().__init__()
        
        # Extract hyperparameters
        # --------------------------------
        densenet_type = "densenet121" if developer_mode else hyperparameters.get("densenet_type", "densenet121")
        if densenet_type == "densenet121":
            block_config = (2, 4, 8, 4) if developer_mode else (6, 12, 24, 16)
        elif densenet_type == "densenet169":
            block_config = (6, 12, 32, 32)
        elif densenet_type == "densenet201":
            block_config = (6, 12, 48, 32)
        else:
            raise ValueError(f"Invalid densenet type: {densenet_type}")

        # Remove last block if:
        # 1. Hyperparameter 'remove_last_block' is explicitly set to True
        # 2. MedMNIST dataset with Baseline run (32x32x32 input too small for 4 blocks)
        #    After conv0 (stride=2) and pool0 (stride=2): 32→16→8, then 3 transitions: 8→4→2→1
        #    With 4 blocks, we get 1x1x1 before denseblock4, which causes InstanceNorm to fail
        should_remove_last_block = (
            hyperparameters.get("remove_last_block", False) or
            (is_medmnist and run_mode == "Baseline")
        )
        
        if should_remove_last_block and len(block_config) == 4:
            block_config = block_config[:3]  # Remove last block: (6, 12, 24, 16) → (6, 12, 24)
            if is_medmnist and run_mode == "Baseline":
                print(f"Reduced DenseNet blocks from 4 to 3 for MedMNIST dataset (32x32x32 input) "
                      f"to prevent InstanceNorm error with 1x1x1 spatial dimensions")
            else:
                print(f"Removed last block from DenseNet as requested by hyperparameter 'remove_last_block'")

        init_features = 8 if developer_mode else hyperparameters.get("init_features", 64)
        growth_rate = 6 if developer_mode else hyperparameters.get("growth_rate", 32)
        bn_size = 1 if developer_mode else hyperparameters.get("bn_size", 4)
        act = hyperparameters.get("act", "relu")
        dropout_prob = hyperparameters.get("dropout_prob", 0.0)
        # --------------------------------
        
        # Build model
        self.model = monai.networks.nets.DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,              # Depends on densenet_type
            bn_size=bn_size,
            act=act,
            norm=("instance", {"affine": True}),    # Fixed for batch_size=1 (added affine=True to ensure weights/bias)
            dropout_prob=dropout_prob
        )
    
    def forward(self, x):
        return self.model(x)


# https://docs.monai.io/en/0.9.1/_modules/monai/networks/nets/resnet.html
# https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/resnet.py#L187-L364
class ParameterizedResNet(nn.Module):
    """
    Parametrized ResNet for 3D medical image classification
    
    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.
        bias_downsample: whether to use bias term in the downsampling block when `shortcut_type` is 'B', default to `True`.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
    
    Note:
        The architecture is compatible with the Med3D pretrained weights: https://github.com/Tencent/MedicalNet
    """
    
    def __init__(self, hyperparameters: dict, num_classes: int, developer_mode: bool):
        super().__init__()
        
        # Extract hyperparameters
        # --------------------------------
        resnet_type = "resnet18" if developer_mode else hyperparameters.get("resnet_type", "resnet18")
        # https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/resnet.py#L49-L58
        if resnet_type == "resnet18":
            block = "basic"
            layers = (2, 2, 2, 2)
            shortcut_type = "A"
            bias_downsample = True
        elif resnet_type == "resnet34":
            block = "basic"
            layers = (3, 4, 6, 3)
            shortcut_type = "A"
            bias_downsample = True
        elif resnet_type == "resnet50":
            block = "bottleneck"
            layers = (3, 4, 6, 3)
            shortcut_type = "B"
            bias_downsample = False
        else:
            raise ValueError(f"Invalid resnet type: {resnet_type}")
        
        conv1_t_size = 3 if developer_mode else hyperparameters.get("conv1_t_size", 7)
        conv1_t_stride = 1 if developer_mode else hyperparameters.get("conv1_t_stride", 1)
        no_max_pool = hyperparameters.get("no_max_pool", False)
        widen_factor = hyperparameters.get("widen_factor", 1.0)
        act = hyperparameters.get("act", "relu")
        # --------------------------------

        # https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/resnet.py#L63-L64
        block_inplanes = [64, 128, 256, 512]
        
        # Build model
        self.model = monai.networks.nets.ResNet(
            block=block,                            # Depends on resnet_type
            layers=layers,                          # Depends on resnet_type
            block_inplanes=block_inplanes,          # Typically fixed to [64, 128, 256, 512]
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_size=conv1_t_size,
            conv1_t_stride=conv1_t_stride,
            no_max_pool=no_max_pool,
            shortcut_type=shortcut_type,            # Depends on resnet_type
            widen_factor=widen_factor,
            num_classes=num_classes,
            feed_forward=True,                      # Fixed to True for classification
            bias_downsample=bias_downsample,        # Depends on resnet_type
            act=act,
            norm=("instance", {"affine": True})     # Fixed for batch_size=1 (added affine=True to ensure weights/bias)
        )
    
    def forward(self, x):
        return self.model(x)


# https://docs.monai.io/en/0.8.0/_modules/monai/networks/nets/efficientnet.html
# https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/efficientnet.py#L478-L562
# NOTE: Using EfficientNetBN instead of EfficientNet to avoid manual block string parsing
class ParameterizedEfficientNetBN(nn.Module):
    """
    Parametrized EfficientNet for 3D medical image classification
    
    Args:
        model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2].
        pretrained: whether to initialize pretrained ImageNet weights, only available for spatial_dims=2 and batch
            norm is used.
        progress: whether to show download progress for pretrained weights download.
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        num_classes: number of output classes.
        norm: feature normalization type and arguments.
        adv_prop: whether to use weights trained with adversarial examples.
            This argument only works when `pretrained` is `True`.

    Note:
        The architecture is NOT compatible with the Med3D pretrained weights: https://github.com/Tencent/MedicalNet
    """
    
    def __init__(self, hyperparameters: dict, num_classes: int):
        super().__init__()
        
        # Extract hyperparameter
        # --------------------------------
        efficientnet_type = hyperparameters.get("efficientnet_type", "efficientnet-b0")
        # --------------------------------

        # Build model
        self.model = monai.networks.nets.EfficientNetBN(
            model_name=efficientnet_type,
            pretrained=False,                       # No pretrained weights for 3D
            progress=False,                         # Not needed when pretrained=False
            spatial_dims=3,
            in_channels=1,
            num_classes=num_classes,
            norm=("instance", {"affine": True}),    # Fixed for batch_size=1 (added affine=True to ensure weights/bias)
            adv_prop=False                          # Not needed when pretrained=False
        )
    
    def forward(self, x):
        return self.model(x)


# https://docs.monai.io/en/1.3.0/_modules/monai/networks/nets/swin_unetr.html
# https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/swin_unetr.py#L45-L330
class ParameterizedSwinUNETR(nn.Module):
    """
    Parametrized SwinUNETR for 3D medical image classification.
    We reuse the SwinUNETR backbone (originally for segmentation) and apply
    global average pooling followed by a linear head for classification.
    
    Args:
        in_channels: dimension of input channels.
        out_channels: dimension of output channels.
        patch_size: size of the patch token.
        feature_size: dimension of network feature size.
        depths: number of layers in each stage.
        num_heads: number of attention heads.
        window_size: local window size.
        qkv_bias: add a learnable bias to query, key, value.
        mlp_ratio: ratio of mlp hidden dim to embedding dim.
        norm_name: feature normalization type and arguments.
        drop_rate: dropout rate.
        attn_drop_rate: attention dropout rate.
        dropout_path_rate: drop path rate.
        normalize: normalize output intermediate features in each stage.
        norm_layer: normalization layer.
        patch_norm: whether to apply normalization to the patch embedding. Default is False.
        use_checkpoint: use gradient checkpointing for reduced memory usage.
        spatial_dims: number of spatial dims.
        downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
            user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
            The default is currently `"merging"` (the original version defined in v0.9.0).
        use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.
    
    Notes:
        - Pretrained weights: Not compatible with Med3D/MedicalNet.
        - Adaptation for classification: We take the backbone output tensor
          with `out_channels=feature_size` and pool over (D, H, W) before a
          linear classifier; no segmentation heads are used.
        - Normalization: InstanceNorm with affine=True is used to be robust with batch_size=1 typical for 3D.
        - What to tune: feature_size, depths, num_heads, window_size, and
          drop rates (drop_rate/attn_drop_rate/dropout_path_rate). Others can
          typically remain at defaults for classification.
    """
    
    def __init__(self, hyperparameters: dict, num_classes: int, developer_mode: bool):
        super().__init__()
        
        # Extract hyperparameters
        # --------------------------------
        # 
        # CONSTRAINTS:
        #
        # 1. spatial_size & patch_size
        # ----------------------------
        # SwinUNETR requires spatial dimensions to be divisible by patch_size * 2^(num_stages-1)
        # With depths=(1,1,1,1) or (2,2,2,2), we have 4 stages, so need divisibility by patch_size * 2^3 = patch_size * 8
        # For spatial_size (64, 64, 32), we need patch_size such that 64 % (patch_size * 8) == 0 and 32 % (patch_size * 8) == 0
        # patch_size=2: 64 % 16 = 0, 32 % 16 = 0
        # patch_size=4: 64 % 32 = 0, 32 % 32 = 0
        # Stage 0: spatial_size / patchsize > e.g., patch_site=4: (64, 64, 32) / 4 = (16, 16, 8)
        # Stage 1: Stage 0 / 2 > e.g., (16, 16, 8) / 2 = (8, 8, 4)
        # Stage 2: Stage 1 / 2 > e.g., (8, 8, 4) / 2 = (4, 4, 2)
        # Stage 3: Stage 2 / 2 > e.g., (4, 4, 2) / 2 = (2, 2, 1)
        # > If each of our spatial dimensions are divisible by 32, we can use a patch_size of 2 and 4
        #
        # 2. window_size
        # ----------------------------
        # Swin window attention operates on the token grid of each stage.
        # Let T_s = spatial_size / (patch_size * 2^s) be the token resolution at stage s (s in {0,1,2,3}).
        # CONSTRAINTS:
        # - window_size must be <= T_s in each dimension (otherwise heavy padding/masking is required).
        # - Ideally, window_size divides T_s with minimal remainder to reduce masks and VRAM.
        # - Anisotropic windows are often best for anisotropic volumes (e.g., Z << X,Y).
        # PRACTICAL EXAMPLE (spatial_size=(416,512,32), patch_size=4):
        # - Tokens per stage: (104,128,8) -> (52,64,4) -> (26,32,2) -> (13,16,1)
        # - Isotropic window_size=7 fits poorly (rarely divides, Z=1 in the last stage).
        # - Good choices: window_size=(4,4,1) or (4,4,2);
        #   * (4,4,1) minimizes Z padding across stages and matches (… , … , 1) at stage 3.
        #   * (4,4,2) is fine up to stage 2 but causes padding/masking at stage 3 (Z=1).
        # RULES OF THUMB:
        # - Choose window_size small enough and such that it divides T_s in as many stages as possible.
        # - Smaller windows and fewer heads save VRAM; larger windows increase cost ~ window_volume * heads.
        # 
        # 3. num_features & num_heads
        # ----------------------------
        # In each attention block must hold: feature_size % num_heads == 0
        # for feature_size f we get stage channels : (f, 2f, 4f, 8f)
        # 
        # Example: For F=24 → stage channels = (24, 48, 96, 192) with num_heads = (3, 6, 12, 24):
        # Each stage's channel count is divisible by its head count.
        # Stage 0: 24 % 3 = 0 > per-head dim = 24/3 = 8
        # Stage 1: 48 % 6 = 0 > per-head dim = 48/6 = 8
        # Stage 2: 96 % 12 = 0 > per-head dim = 96/12 = 8
        # Stage 3: 192 % 24 = 0 > per-head dim = 192/24 = 8

        # Convert depths_0 to tuple if present, otherwise use depths directly
        if developer_mode:
            depths_0 = 1
        else:
            depths_0 = hyperparameters.get("depths_0", 2)
            depths = (depths_0, depths_0, depths_0, depths_0)
        
        # Convert num_heads per stage to tuple if present, otherwise use num_heads directly
        if developer_mode:
            num_heads = (1, 2, 3, 4)
        elif all(f"num_heads_{i}" in hyperparameters for i in range(4)):
            num_heads = (
                hyperparameters.get("num_heads_0", 3),
                hyperparameters.get("num_heads_1", 6),
                hyperparameters.get("num_heads_2", 12),
                hyperparameters.get("num_heads_3", 24),
            )
        else:
            num_heads = hyperparameters.get("num_heads", (3, 6, 12, 24))
        
        # NOTE: window_size is fixed to (4, 4, 1) to minimize Z padding across stages
        # This works well for anisotropic medical 3D volumes (Z << X, Y)
        # Parametrization is NOT recommended due to complex constraints:
        # - window_size must be <= token_resolution at each stage
        # - Token resolution depends on spatial_size and patch_size (dataset-dependent)
        # - (4, 4, 1) minimizes padding/masking and matches stage 3 where Z often = 1

        # NOTE: feature_size is fixed to 24 for good divisibility and to satisfy the constraint: (feature_size * 2^s) % num_heads[s] == 0
        # Higher feature_sizes (like 48 which is also divisible by 24) is not feasible due to memory reasons

        patch_size = 2 if developer_mode else hyperparameters.get("patch_size", 2)  # larger patches = fewer tokens
        feature_size = 12 if developer_mode else 24
        depths = (depths_0, depths_0, depths_0, depths_0)  # default: (2, 2, 2, 2)
        num_heads = num_heads  # default: (3, 6, 12, 24)
        window_size = 2 if developer_mode else (4, 4, 1)  # Update from 7 to (4,4,1): minimizes Z padding across stages and matches (… , … , 1) at stage 3
        mlp_ratio = 2.0 if developer_mode else hyperparameters.get("mlp_ratio", 4.0)
        drop_rate = hyperparameters.get("dropout_prob", 0.0)
        attn_drop_rate = hyperparameters.get("attn_drop_rate", 0.0)
        dropout_path_rate = hyperparameters.get("dropout_path_rate", 0.0)

        # Validate constraint: (feature_size * 2^s) % num_heads[s] == 0 for each stage
        stage_channels = [feature_size * (2 ** s) for s in range(4)]
        for s, (channels, heads) in enumerate(zip(stage_channels, num_heads)):
            if channels % heads != 0:
                raise ValueError(
                    f"Constraint violation at stage {s}: "
                    f"channels={channels} (feature_size={feature_size} * 2^{s}) "
                    f"must be divisible by num_heads={heads}, but {channels} % {heads} = {channels % heads}"
                )
        # --------------------------------
        
        # Build model
        # Extract hyperparameters with sensible defaults
        self.backbone = monai.networks.nets.SwinUNETR(
            in_channels=1,
            out_channels=feature_size,                 # Set to feature_size; pooled for classification
            patch_size=patch_size,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=True,                             # Default setting works well for classification
            mlp_ratio=mlp_ratio,
            norm_name=("instance", {"affine": True}),  # Fixed for batch_size=1 (added affine=True to ensure weights/bias)
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=False,                           # Intermediate feature norm not needed for classification
            patch_norm=False,                          # Patch embedding normalization not required here
            use_checkpoint=True,                      # Enable to save VRAM if needed (no quality impact)
            spatial_dims=3,
            downsample="merging",                      # Default; minor effect on capacity/compute
            use_v2=False                               # Default; optional residual conv at stage starts
        )

        # Classification head: global pooling + linear
        self.global_pool = nn.AdaptiveAvgPool3d(1)     # [B, C, D, H, W] -> [B, C, 1, 1, 1]
        self.classifier  = nn.Linear(feature_size, num_classes)
    
    def forward(self, x):
        # Full SwinUNETR forward returns a dense feature map with 'out_channels' channels.
        # Shape: [B, feature_size, D, H, W]
        feat = self.backbone(x)

        # Global average pooling over (D, H, W)
        pooled = self.global_pool(feat).flatten(1)  # [B, feature_size]

        # Linear classifier
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


# https://docs.monai.io/en/1.3.0/_modules/monai/networks/nets/vit.html
# https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/vit.py#L25-L133
class ParameterizedViT(nn.Module):
    """
    Parametrized Vision Transformer for 3D medical image classification
    
    Args:
        in_channels (int): dimension of input channels.
        img_size (Union[Sequence[int], int]): dimension of input image.
        patch_size (Union[Sequence[int], int]): dimension of patch size.
        hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
        mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
        num_layers (int, optional): number of transformer blocks. Defaults to 12.
        num_heads (int, optional): number of attention heads. Defaults to 12.
        proj_type (str, optional): patch embedding layer type. Defaults to "conv".
        pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
        classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
        num_classes (int, optional): number of classes if classification is used. Defaults to 2.
        dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
        spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
        post_activation (str, optional): add a final acivation function to the classification head
            when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
            Set to other values to remove this function.
        qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
        save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.
    
    Note:
        The architecture is NOT compatible with the Med3D pretrained weights: https://github.com/Tencent/MedicalNet
    """
    
    def __init__(self, hyperparameters: dict, num_classes: int, developer_mode: bool, spatial_size: tuple):
        super().__init__()
        
        print(f"\nSpatial size at ViT: {spatial_size}\n")
        
        # Extract hyperparameters
        # --------------------------------
        #
        # CONSTRAINTS:
        #
        # 1. spatial_size & patch_size
        # ----------------------------
        # img_size[i] % patch_size[i] == 0 for all i
        #
        # Example: spatial_size = (416, 512, 32) and patch_size = (8, 8, 4)
        # > tokens = (416/8, 512/8, 32/4) = (52, 64, 8); sequence length = 52 * 64 * 8 = 26624
        # > With smaller patch_size, we get more tokens
        # > More tokens > Higher N > Memory scales roughly with N² per layer (self-attention), so  is the main VRAM driver
        #
        # 2. hidden_size & num_heads
        # ----------------------------
        # hidden_size % num_heads == 0
        #
        # Example: hidden_size = 768 and num_heads = 12
        # > hidden_size % num_heads = 768 % 12 = 0
        #
        # 3. mlp_dim & hidden_size
        # ----------------------------
        # Usual rule: mlp_dim = 2 * hidden_size or mlp_dim = 4 * hidden_size
        # (2x is memory friendlier, 4x is stronger)
        #
        # Example: mlp_dim = 3072 or 1536 and hidden_size = 768
        # hidden_size * 2 = 768 * 2 = 1536
        # hidden_size * 4 = 768 * 4 = 3072
        #
        # 4. num_layers
        # ----------------------------
        # More num_layers > More parameters > More VRAM

        patch_size_0 = hyperparameters.get("patch_size_0", 16)
        patch_size = (patch_size_0, patch_size_0, patch_size_0 // 2)  # Increased from (8, 8, 4) to (16, 16, 8) due to memory reasons
        hidden_size = 96 if developer_mode else int(12 * hyperparameters.get("hidden_size_multiplier", 64))  # 12 * 64 = 768
        mlp_dim = 1536 if developer_mode else (hidden_size * hyperparameters.get("mlp_dim_multiplier", 4))  # 768 * 4 = 3072
        num_layers = 1 if developer_mode else hyperparameters.get("num_layers", 12)
        num_heads = 3 if developer_mode else hyperparameters.get("num_heads", 6)  # Decreased from 12 to 6 due to memory reasons
        pos_embed_type = hyperparameters.get("pos_embed_type", "learnable")
        dropout_prob = hyperparameters.get("dropout_prob", 0.0)
        qkv_bias = hyperparameters.get("qkv_bias", False)
        # --------------------------------
    
        # Build model
        self.model = monai.networks.nets.ViT(
            in_channels=1,
            img_size=spatial_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type="conv",               # keep default; robust in 3D
            pos_embed_type=pos_embed_type,
            classification=True,            # Enable classification mode
            num_classes=num_classes,
            dropout_rate=dropout_prob,
            spatial_dims=3,
            post_activation="",             # Use "" so that the classification loggits are unclipped for CrossEntropyLoss
            qkv_bias=qkv_bias,
            save_attn=False,                # keep default; just relevant for visualization, debugging
            # NOTE: LayerNorm implemented for Transformer architecture compatibility. It is batch_size agnostic.
        )
    
    def forward(self, x):
        output = self.model(x)
        # MONAI ViT might return a tuple, we need only the classification output
        if isinstance(output, tuple):
            return output[0]  # Return only the classification logits
        return output