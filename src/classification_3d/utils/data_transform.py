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
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    RandRotated,
    RandZoomd,
    RandFlipd,
    ResizeWithPadOrCropd,
    # nnU-Net augmentation transforms (corrected names for MONAI 1.5.0)
    RandAffined,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandBiasFieldd,
    RandCoarseDropoutd,
)
from monai.data import Dataset
from torch.utils.data import DataLoader

def DataTransform(normalization_stats, developer_mode, spatial_size=None, is_training=True):
    """
    Transform the training, validation, and test data. For training set, it applies data augmentation.
    
    Args:
        normalization_stats (dict): Normalization statistics for CT images only; default None for MRI images
        developer_mode (bool): If True, uses smaller model target shape for faster development; default False
        spatial_size (tuple): Image size in (D, H, W) format for e.g., ViT; default None
        is_training (bool): If True, applies data augmentation; if False, no data augmentation; default True
    """
    # Base transform that is always applied
    transforms = [
        LoadImaged(keys="image", image_only=True),  # Load NIfTI images
    ]

    # Add normalization for CT images only
    is_mri = normalization_stats is None
    if not is_mri:
        transforms.append(
            NormalizeIntensityd(keys=["image"], subtrahend=normalization_stats["mean"][0], divisor=normalization_stats["std"][0])
        )

    # Add resizing if needed:
    # > For developer mode, we need to resize the image to a smaller spatial size in the format (D, H, W) for faster development.
    # > For some models e.g., ViT, we need to resize the image to a specific spatial size in the format (D, H, W).
    needs_resizing = developer_mode or spatial_size is not None
    if needs_resizing:
        transforms.append(
            ResizeWithPadOrCropd(keys="image", spatial_size=spatial_size, mode="constant", constant_values=0)
        )
    
    # Add data augmentation for training set only; no data augmentation for validation and test set!
    if is_training:
        transforms.append(
            RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
            RandRotated(keys=["image"], range_z=(-25, 25), prob=0.2),
            RandZoomd(keys=["image"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
        )
    
    return Compose(transforms)


def NNUNETAugmentTransform(voxel_size, image_size, normalization_stats, developer_mode):
    """
    Transform for training on the training set with nnU-Net-inspired data augmentation strategy.
    
    This function implements a comprehensive augmentation pipeline based on nnU-Net's approach,
    which includes elastic deformation, gamma correction, rotation, scaling, mirroring, and
    intensity variations to improve model robustness and generalization.
    
    Args:
        voxel_size (tuple): Voxel size in (x, y, z) format
        image_size (tuple): Image size in (H, W, D) format for ViT; default None
        normalization_stats (dict): Normalization statistics
        developer_mode (bool): If True, uses smaller model target shape for faster development

    Returns:
        monai.transforms.Compose: Compose object containing the transformations
    """
    # Define augmentation probabilities - can be made configurable later
    augmentation_prob = 0.3  # Probability for each augmentation

    print(f"\n\n\nUsing nnU-Net Augmentation Transform with probability {augmentation_prob}\n\n\n")
    
    # Determine if this is MRI or CT dataset
    is_mri = normalization_stats is None
    needs_resizing = developer_mode or image_size is not None
    
    # Base transforms that are always applied
    transforms = [
        LoadImaged(keys="image", image_only=True),  # Load NIfTI images
        EnsureChannelFirstd(keys="image"),  # Ensure channels are first (for compatibility)
        Spacingd(keys="image", pixdim=voxel_size, mode="bilinear"),  # Resample to target spacing
    ]
    
    # Add normalization for CT images only
    if not is_mri:
        transforms.append(
            NormalizeIntensityd(keys=["image"], subtrahend=normalization_stats["mean"][0], divisor=normalization_stats["std"][0])
        )
    
    # Add resizing if needed
    if needs_resizing:
        transforms.append(
            ResizeWithPadOrCropd(keys="image", spatial_size=image_size, mode="constant", constant_values=0)
        )
    
    # Define augmentation parameters based on modality
    if is_mri:
        # MRI Images: gist, crlm, melanoma - more aggressive intensity augmentations
        gamma_range = (0.7, 1.5)
        intensity_factors = 0.1
        intensity_offsets = 0.1
        noise_std = 0.01
        bias_field_coeff = (0.0, 0.3)
    else:
        # CT Images: lipo, desmoid, liver - more conservative intensity augmentations
        gamma_range = (0.8, 1.2)
        intensity_factors = 0.05
        intensity_offsets = 0.05
        noise_std = 0.005
        bias_field_coeff = (0.0, 0.2)
    
    # nnU-Net inspired data augmentation pipeline
    augmentation_transforms = [
        # 1. Spatial augmentations (applied with higher probability for geometric robustness)
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # Left-right flip
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # Anterior-posterior flip  
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),  # Superior-inferior flip
        
        # 2. Rotation augmentation (nnU-Net uses random rotations)
        RandRotated(keys=["image"], range_x=(-15, 15), prob=augmentation_prob),
        RandRotated(keys=["image"], range_y=(-15, 15), prob=augmentation_prob),
        RandRotated(keys=["image"], range_z=(-15, 15), prob=augmentation_prob),
        
        # 3. Scaling augmentation (nnU-Net uses random scaling)
        RandZoomd(keys=["image"], prob=augmentation_prob, min_zoom=0.85, max_zoom=1.15),
        
        # 4. Elastic deformation (key nnU-Net augmentation)
        Rand3DElasticd(
            keys=["image"], 
            prob=augmentation_prob,
            sigma_range=(3, 7),  # Control deformation smoothness
            magnitude_range=(50, 150),  # Control deformation strength
            spatial_size=image_size if needs_resizing else None,
            mode="bilinear",
            padding_mode="border"
        ),
        
        # 5. Affine transformation (combines rotation, translation, scaling)
        RandAffined(
            keys=["image"],
            prob=augmentation_prob,
            translate_range=(10, 10, 5),  # Translation in x, y, z
            rotate_range=(0.1, 0.1, 0.1),  # Rotation in radians
            scale_range=(0.1, 0.1, 0.1),  # Scaling factors
            spatial_size=image_size if needs_resizing else None,
            mode="bilinear",
            padding_mode="border"
        ),
        
        # 6. Intensity augmentations (nnU-Net uses gamma correction and noise)
        RandAdjustContrastd(keys=["image"], prob=augmentation_prob, gamma=gamma_range),  # Gamma correction
        RandScaleIntensityd(keys=["image"], prob=augmentation_prob, factors=intensity_factors),  # Intensity scaling
        RandShiftIntensityd(keys=["image"], prob=augmentation_prob, offsets=intensity_offsets),  # Intensity shift
        RandGaussianNoised(keys=["image"], prob=augmentation_prob, std=noise_std),  # Gaussian noise
        
        # 7. Additional augmentations for robustness
        RandGaussianSmoothd(keys=["image"], prob=augmentation_prob*0.5, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        RandBiasFieldd(keys=["image"], prob=augmentation_prob*0.3, coeff_range=bias_field_coeff),  # Bias field simulation
        RandCoarseDropoutd(keys=["image"], prob=augmentation_prob*0.2, holes=3, spatial_size=(8, 8, 4)),  # Random dropout
    ]
    
    # Combine base transforms with augmentation transforms
    transforms.extend(augmentation_transforms)
    
    return Compose(transforms)