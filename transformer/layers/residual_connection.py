from torch import Tensor
import torch.nn as nn

from transformer.layers.layer_norm import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float = 0.1):
        """
        Args:
            features (int): feature dimensions
            dropout (float): dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """
        Args:
            x (Tensor): input tensor, shape ``(batch_size, seq_length, d_model)``
            sublayer (nn.Module): sublayer module

        Returns:
            output (Tensor): shape ``(batch_size, seq_length, d_model)``
        """
        return x + self.dropout(sublayer(self.norm(x)))
        # return self.norm(x + self.dropout(sublayer(x)))
