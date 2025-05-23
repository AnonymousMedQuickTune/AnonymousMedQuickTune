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

# DenseNet model from MONAI
class DenseModel(nn.Module):
    
    def __init__(
        self,
        hyperparameters, 
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act = "relu",
        norm = "instance",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(in_channels, hyperparameters["init_features"], kernel_size=5, # kernel size changed from 7 to 5.
                                        stride=hyperparameters["conv0_stride"], # This will become dynamic. (Before 2)
                                        #stride=2, # This will become dynamic. (Before 2)
                                        padding=3, # Padding changed from 3 to 2.
                                        bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=hyperparameters["init_features"])),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)), # stride before 2
                ]
            )
        )

        in_channels = hyperparameters["init_features"]
        block_config = (hyperparameters["num_layers_block1"], hyperparameters["num_layers_block2"], hyperparameters["num_layers_block3"], hyperparameters["num_layers_block4"])
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=hyperparameters["bn_size"],
                growth_rate=hyperparameters["growth_rate"],
                dropout_prob=hyperparameters["dropout_prob"],
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * hyperparameters["growth_rate"]
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", nn.AdaptiveAvgPool3d(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x
    
class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act = "relu",
        norm = "instance",
    ) -> None:

        super().__init__()

        out_channels = bn_size * growth_rate

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", nn.Conv3d(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", nn.Dropout(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act = "relu",
        norm = "instance",
    ) -> None:
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act = "relu",
        norm = "instance",
    ) -> None:

        super().__init__()

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


# DenseNet model from MONAI
class DenseNetNoConfig(nn.Module):
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act = "relu",
        norm = "instance",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(in_channels, init_features, kernel_size=5, # kernel size changed from 7 to 5.
                                        #stride=config["conv0_stride"], # This will become dynamic. (Before 2)
                                        stride=2, # This will become dynamic. (Before 2)
                                        padding=3, # Padding changed from 3 to 2.
                                        bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)), # stride before 2
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", nn.AdaptiveAvgPool3d(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x