import torch.nn as nn
from torch import Tensor

from transformer.layers import (
    MultiHeadAttention,
    ResidualConnection,
    PositionWiseFeedForward,
    LayerNormalization,
)
from transformer.utils.functional import get_clones

class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            num_heads (int): number of attention heads
            d_ffn (int): dimension of feed-forward network
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate=attention_dropout_rate)
        self.attention_residual_connection = ResidualConnection(d_model, dropout_rate=dropout_rate)

        self.position_wise_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout_rate=dropout_rate)
        self.ffn_residual_connection = ResidualConnection(d_model, dropout_rate=dropout_rate)

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
    def __init__(
        self,
        encoder_layer: EncoderLayer,
        d_model: int,
        num_layers: int,
    ):
        """
        Args:
            encoder_layer (EncoderLayer): encoder layer
            d_model (int): dimension of the embedding vectors
            num_layers (int): number of encoder layers
        """
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.norm = LayerNormalization(d_model)

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
