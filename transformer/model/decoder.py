import torch.nn as nn
from torch import Tensor

from ..layers import (
    MultiHeadAttention,
    ResidualConnection,
    PositionWiseFeedForward,
    LayerNormalization,
)
from ..utils.functional import get_clones

class DecoderLayer(nn.Module):
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
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout_rate=attention_dropout_rate)
        self.masked_attention_residual_connection = ResidualConnection(d_model, dropout_rate=dropout_rate)

        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_rate=attention_dropout_rate)
        self.cross_attention_residual_connection = ResidualConnection(d_model, dropout_rate=dropout_rate)

        self.position_wise_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout_rate=dropout_rate)
        self.ffn_residual_connection = ResidualConnection(d_model, dropout_rate=dropout_rate)

    def forward(
        self,
        src: Tensor,
        target: Tensor,
        src_mask: Tensor,
        target_mask: Tensor
    ) -> Tensor:
        """
        Args:
            src (Tensor): output from encoder, shape ``(batch_size, src_seq_length, d_model)``
            target (Tensor): target tensor, shape ``(batch_size, target_seq_length, d_model)``
            src_mask (Tensor): source mask tensor, shape ``(batch_size, 1, 1, src_seq_length)``
            target_mask (Tensor): target mask tensor, shape ``(batch_size, 1, target_seq_length, target_seq_length)``

        Returns:
            target (Tensor): sequences after self-attention, shape ``(batch_size, target_seq_length, d_model)``
        """

        # passing through multi head attention layer
        target = self.masked_attention_residual_connection(
            target,
            lambda x: self.masked_attention(x, x, x, mask=target_mask)
        )

        # passing through multi head cross attention layer
        target = self.cross_attention_residual_connection(
            target,
            lambda x: self.cross_attention(x, src, src, mask=src_mask)
        )

        # passing through position wise feed forward network layer
        target = self.ffn_residual_connection(
            target,
            self.position_wise_ffn
        )
        return target

class Decoder(nn.Module):
    def __init__(
        self,
        decoder_layer: DecoderLayer,
        d_model: int,
        num_layers: int,
    ):
        """
        Args:
            decoder_layer (DecoderLayer): decoder layer
            d_model (int): dimension of the embedding vectors
            num_layers (int): number of decoder layers
        """
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.norm = LayerNormalization(d_model)

    def forward(
        self,
        src: Tensor,
        target: Tensor,
        src_mask: Tensor,
        target_mask: Tensor
    ) -> Tensor:
        """
        Args:
            src (Tensor): output from encoder, shape ``(batch_size, src_seq_length, d_model)``
            target (Tensor): target tensor, shape ``(batch_size, target_seq_length, d_model)``
            src_mask (Tensor): source mask tensor, shape ``(batch_size, 1, 1, src_seq_length)``
            target_mask (Tensor): target mask tensor, shape ``(batch_size, 1, target_seq_length, target_seq_length)``

        Returns:
            target (Tensor): target after self-attention, shape ``(batch_size, target_seq_length, d_model)``
        """
        for layer in self.layers:
            target = layer(src, target, src_mask, target_mask)

        target = self.norm(target)
        return target
