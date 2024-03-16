import math

import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int, dropout_rate: float = 0.1):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            max_seq_length (int): maximum length of the sequences
            dropout_rate (float): randomly zeroes-out some of the input
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros((max_seq_length, d_model))
        positions = torch.arange(max_seq_length, dtype=torch.float).unsqueeze(1)  # (max_seq_length, 1)
        div_term = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float) * math.log(10000.0) / d_model)

        # calculate sine for even indices
        pe[:, ::2] = torch.sin(positions * div_term)

        # calculate cos for odd indices
        pe[:, 1::2] = torch.cos(positions * div_term)

        # add a dimension to compatible with batch dimension
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): embedded inputs, shape ``(batch_size, seq_length, d_model)``

        Returns:
            x + positional encoding, shape ``(batch_size, seq_length, d_model)``
        """
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        x = self.dropout(x)
        return x
