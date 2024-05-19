from torch import Tensor
import torch.nn as nn

from transformer.layers import (
    MultiHeadAttention,
    ResidualConnection,
    PositionWiseFeedForward,
    LayerNormalization,
)
from transformer.model.transformer_config import TransformerConfig

class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config.d_model, config.num_heads, dropout=config.attention_dropout)
        self.attention_residual_connection = ResidualConnection(config.d_model, dropout=config.dropout)

        self.position_wise_ffn = PositionWiseFeedForward(config.d_model, config.d_ffn, dropout=config.dropout)
        self.ffn_residual_connection = ResidualConnection(config.d_model, dropout=config.dropout)

    def forward(self, inputs: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            inputs (Tensor): positionally embedded inputs tensor, shape ``(batch_size, seq_length, d_model)``
            src_mask (Tensor): mask tensor, shape ``(batch_size, 1, 1, seq_length)`` (default: None)

        Returns:
            output (Tensor): sequences after a single self-attention layer, shape ``(batch_size, seq_length, d_model)``
        """

        # passing through multi head attention layer
        inputs = self.attention_residual_connection(
            inputs,
            lambda x: self.attention(x, x, x, mask=src_mask)
        )

        # passing through position-wise feed-forward network
        inputs = self.ffn_residual_connection(
            inputs,
            self.position_wise_ffn
        )
        return inputs

class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = LayerNormalization(config.d_model)

    def forward(self, inputs: Tensor, src_mask: Tensor | None = None):
        """
        Args:
            inputs (Tensor): positionally embedded inputs tensor, shape ``(batch_size, seq_length, d_model)``
            src_mask (Tensor): mask tensor, shape ``(batch_size, 1, 1, seq_length)`` (default: None)

        Returns:
            output (Tensor): sequences after self-attention ``(batch_size, seq_length, d_model)``
        """
        for layer in self.layers:
            inputs = layer(inputs, src_mask=src_mask)

        inputs = self.norm(inputs)
        return inputs
