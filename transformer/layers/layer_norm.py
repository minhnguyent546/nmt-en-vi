import torch
import torch.nn as nn
from torch import Tensor

class LayerNormalization(nn.Module):
    def __init__(self, features, eps: float = 1e-7):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor, shape ``(batch_size, seq_length, d_model)``
        Returns:
            y (Tensor): standardized tensor, shape ``(batch_size, seq_length, d_model)``
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()  # prevent std become too large when var is ~ zero
        y = (x - mean) / std
        y = self.gamma * y + self.beta
        return y
