import torch.nn as nn
from torch import Tensor

from .layer_norm import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout_rate: float = 0.1):
        """
        Args:
            features (int): feature dimensions
            dropout_rate (float): dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = LayerNormalization(features)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """
        Args:
            x (Tensor): input tensor, shape ``(batch_size, seq_length, d_model)``
            sublayer (nn.Module): sublayer module

        Returns:
            output (Tensor): shape ``(batch_size, seq_length, d_model)``
        """
        # return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))
