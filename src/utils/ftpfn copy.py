# NePS FTPFN: https://github.com/automl/neps/blob/master/neps/optimizers/models/ftpfn.pyf
# NePS GP: https://github.com/automl/neps/blob/master/neps/optimizers/models/gp.py

from __future__ import annotations

import torch
from dataclasses import dataclass
from pathlib import Path
import torch.nn.functional as F

from qtt.predictors.models import FeatureEncoder
from ifbo.utils import Curve
from ifbo.surrogate import FTPFN

class FTPFNSurrogateModel(torch.nn.Module):
    """Combines QuickTune's encoder architecture with IfBO's FT-PFN model."""
    
    def __init__(
        self,
        in_dim: int | list[int],
        in_curve_dim: int,
        out_dim: int = 32,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 32,
        enc_nlayers: int = 3,
        out_curve_dim: int = 16,
        target_path: Path | str | None = None,
        version: str = "0.0.1",
        device: torch.device | None = None,
    ):
        super().__init__()
        
        # Store device and dimensions
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_dim = in_dim[0] if isinstance(in_dim, list) else in_dim
        
        # QuickTune's feature encoder
        self.encoder = FeatureEncoder(
            in_dim,
            in_curve_dim,
            out_dim,
            enc_hidden_dim,
            enc_out_dim,
            enc_nlayers,
            out_curve_dim,
        ).to(self.device)
        
        # IfBO's FT-PFN model
        self.ftpfn = FTPFN(
            target_path=target_path,
            version=version,
            device=self.device
        )
        
        # Dummy attributes for QuickTune compatibility
        self.lengthscale = torch.tensor([1.0], device=self.device)
        self.noise = torch.tensor([0.1], device=self.device)
        
    def _normalize_data(self, pipeline, curve, y=None):
        """Normalize input data to [0,1] range as required by FTPFN."""
        # Normalize pipeline features
        pipeline_min = pipeline.min(dim=0)[0]
        pipeline_max = pipeline.max(dim=0)[0]
        pipeline_norm = (pipeline - pipeline_min) / (pipeline_max - pipeline_min + 1e-8)
        
        # Normalize learning curves
        curve_min = curve.min(dim=0)[0]
        curve_max = curve.max(dim=0)[0]
        curve_norm = (curve - curve_min) / (curve_max - curve_min + 1e-8)
        
        # Normalize targets if provided
        y_norm = None
        if y is not None:
            y_min = y.min()
            y_max = y.max()
            y_norm = (y - y_min) / (y_max - y_min + 1e-8)
            
        return pipeline_norm, curve_norm, y_norm

    def _check_input(self, pipeline, curve, y=None):
        """Validate input dimensions and values."""
        if pipeline.dim() != 2:
            raise ValueError(f"Pipeline should be 2D, got shape {pipeline.shape}")
        if curve.dim() != 2:
            raise ValueError(f"Curve should be 2D, got shape {curve.shape}")
        if y is not None and y.dim() != 1:
            raise ValueError(f"Target should be 1D, got shape {y.shape}")
            
        if pipeline.size(0) != curve.size(0):
            raise ValueError(f"Number of samples mismatch: {pipeline.size(0)} vs {curve.size(0)}")
        if y is not None and pipeline.size(0) != y.size(0):
            raise ValueError(f"Number of samples mismatch: {pipeline.size(0)} vs {y.size(0)}")

    def forward(self, pipeline, curve):
        # Move input tensors to correct device and enable gradient tracking
        pipeline = pipeline.to(self.device).requires_grad_(True)
        curve = curve.to(self.device).requires_grad_(True)
        
        # Input validation
        self._check_input(pipeline, curve)
        
        # Normalize inputs
        pipeline_norm, curve_norm, _ = self._normalize_data(pipeline, curve)
        
        # Encode features using QuickTune's encoder
        encoded_features = self.encoder(pipeline_norm, curve_norm)
        
        # Normalize encoded features and ensure gradient tracking
        encoded_features_min = encoded_features.min(dim=0)[0]
        encoded_features_max = encoded_features.max(dim=0)[0]
        encoded_features_norm = ((encoded_features - encoded_features_min) / 
                               (encoded_features_max - encoded_features_min + 1e-8)).requires_grad_(True)
        
        # Reduce feature dimension to 8 (FTPFN's expected feature size)
        if encoded_features_norm.size(1) > 8:
            encoded_features_norm = encoded_features_norm[:, :8].requires_grad_(True)
        elif encoded_features_norm.size(1) < 8:
            padding = torch.zeros(encoded_features_norm.size(0), 
                                8 - encoded_features_norm.size(1), 
                                device=self.device,
                                requires_grad=True)
            encoded_features_norm = torch.cat([encoded_features_norm, padding], dim=1)
        
        # Convert to IfBO's format
        context_curves = []
        query_curves = []
        
        n_samples = pipeline.size(0)
        # Normalize ID to [0,1] range as required by IfBO FTPFN
        # All hyperparameter values must be in [0,1]
        max_id = max(1, n_samples - 1)  # Avoid division by zero
        
        # Create curves with proper ID and timestep format
        for i, (features, curve) in enumerate(zip(encoded_features_norm, curve_norm)):
            # Normalize ID to [0,1] range
            normalized_id = float(i) / max_id if max_id > 0 else 0.0
            normalized_id = min(1.0, normalized_id)  # Ensure it's at most 1.0
            
            # Add ID and timestep to features
            features_with_meta = torch.cat([
                torch.tensor([normalized_id], device=self.device),  # ID normalized to [0,1]
                torch.zeros(1, device=self.device),  # Timestep placeholder
                features  # Encoded features already in [0,1]
            ])
            
            # During training: create curves for each timestep
            if hasattr(self, 'train_pipeline'):
            for t in range(curve.size(0)):
                    timestep = float(t) / curve.size(0) if curve.size(0) > 0 else 0.0
                    # Create a copy to avoid modifying the original tensor
                features_for_timestep = features_with_meta.clone()
                features_for_timestep[1] = timestep
                curve_value = curve[t].unsqueeze(0).unsqueeze(0)  # Make 2D tensor [1,1]
                
                    # Use current curve as query, all others as context
                    if i == len(encoded_features_norm) - 1:
                        query_curves.append(Curve(
                            hyperparameters=features_for_timestep,
                            t=torch.tensor([[timestep]], device=self.device),  # Make 2D tensor [1,1]
                            y=None
                        ))
                    else:
                        context_curves.append(Curve(
                            hyperparameters=features_for_timestep,
                            t=torch.tensor([[timestep]], device=self.device),  # Make 2D tensor [1,1]
                            y=curve_value
                        ))
            # During inference: only use the last timestep for prediction (one prediction per config)
                else:
                # Use the last timestep (final performance) for prediction
                last_t = curve.size(0) - 1
                timestep = 1.0 if curve.size(0) > 0 else 0.0  # Last timestep normalized to 1.0
                features_for_timestep = features_with_meta.clone()
                features_for_timestep[1] = timestep
                curve_value = curve[last_t].unsqueeze(0).unsqueeze(0)  # Make 2D tensor [1,1]
                
                    if i == 0:
                    # First config: use as context
                        context_curves.append(Curve(
                            hyperparameters=features_for_timestep,
                            t=torch.tensor([[timestep]], device=self.device),  # Make 2D tensor [1,1]
                            y=curve_value
                        ))
                    else:
                    # Other configs: use as query (one per config)
                        query_curves.append(Curve(
                            hyperparameters=features_for_timestep,
                            t=torch.tensor([[timestep]], device=self.device),  # Make 2D tensor [1,1]
                            y=None
                        ))
        
        # Ensure we have at least one context curve
        if not context_curves:
            # Use the first query curve as context
            first_query = query_curves[0]
            context_curves.append(Curve(
                hyperparameters=first_query.hyperparameters,
                t=first_query.t,
                y=curve_norm[0][0].unsqueeze(0).unsqueeze(0).to(self.device)  # Make 2D tensor [1,1]
            ))
        
        # Ensure we have at least one query curve
        if not query_curves:
            # Use the last context curve as query
            last_context = context_curves[-1]
            query_curves.append(Curve(
                hyperparameters=last_context.hyperparameters,
                t=last_context.t,
                y=None
            ))
        
        # Get predictions using FT-PFN and ensure gradients are tracked
        with torch.set_grad_enabled(self.training):
            predictions = self.ftpfn.predict(context_curves, query_curves)
            logits = predictions[0].logits.to(self.device)
            # Ensure predictions have the right shape
            # During inference, we should have one prediction per query curve (one per config)
            mean = logits.squeeze()
            if mean.dim() == 0:
                mean = mean.unsqueeze(0)
            
            # Ensure we have exactly n_samples predictions (one per configuration)
            # If we have more predictions than samples, take only the first n_samples
            # This can happen if context curves are also included in predictions
            if mean.size(0) > n_samples:
                mean = mean[:n_samples]
            elif mean.size(0) < n_samples:
                # If we have fewer predictions, pad with the last prediction
                last_pred = mean[-1:] if mean.size(0) > 0 else torch.zeros(1, device=self.device)
                padding = last_pred.repeat(n_samples - mean.size(0))
                mean = torch.cat([mean, padding])
            
            # FT-PFN doesn't provide explicit uncertainty, so we use a small default stddev
            # This is a reasonable default for normalized predictions in [0,1] range
            stddev = torch.full_like(mean, 0.1)  # Default uncertainty of 0.1
            return PredictionOutput(mean=mean, stddev=stddev)

    def train_step(self, pipeline, curve, y):
        """Training step required by QuickTune API."""
        # Move all inputs to correct device and enable gradient tracking
        pipeline = pipeline.to(self.device).requires_grad_(True)
        curve = curve.to(self.device).requires_grad_(True)
        y = y.to(self.device).requires_grad_(True)
        
        # Input validation
        self._check_input(pipeline, curve, y)
        
        # Store training data
        self.train_pipeline = pipeline
        self.train_curve = curve
        self.train_y = y
        
        # Make prediction and compute loss
        predictions = self(pipeline, curve)
        y_norm = self._normalize_data(pipeline, curve, y)[2]
        
        # Ensure predictions and target have same shape
        mean = predictions.mean
        if mean.size(0) != y_norm.size(0):
            mean = mean[-y_norm.size(0):]
        
        # Compute loss with gradient tracking
        loss = F.mse_loss(mean, y_norm)
        return loss

    @torch.no_grad()
    def predict(self, pipeline, curve):
        """Prediction method required by QuickTune API."""
        # Move inputs to correct device
        pipeline = pipeline.to(self.device)
        curve = curve.to(self.device)
        return self(pipeline, curve)

    @torch.no_grad()
    def set_train_data(self, pipeline, curve, y):
        """Set training data (required by QuickTune API)."""
        # Move all inputs to correct device
        pipeline = pipeline.to(self.device)
        curve = curve.to(self.device)
        y = y.to(self.device)
        
        self._check_input(pipeline, curve, y)
        self.eval()
        self.train_pipeline = pipeline
        self.train_curve = curve
        self.train_y = y

@dataclass
class PredictionOutput:
    """Container for prediction output to match QuickTune API"""
    mean: torch.Tensor
    stddev: torch.Tensor


