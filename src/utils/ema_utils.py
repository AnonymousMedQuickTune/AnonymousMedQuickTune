import copy
import torch

class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    Keeps a shadow copy of the model and updates it smoothly over time.
    """
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model).eval()  # clone model
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update EMA weights from the current model."""
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def to(self, device):
        """Move EMA model to device."""
        self.ema_model.to(device)
