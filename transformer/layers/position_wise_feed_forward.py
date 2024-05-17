import torch.nn as nn
from torch import Tensor

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            d_ffn (int): dimension of the hidden layer
            dropout (float): dropout rate
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn)
        self.linear_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor, shape ``(batch_size, seq_length, d_model)``

        Returns:
            output (Tensor): shape ``(batch_size, seq_length, d_model)``
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
