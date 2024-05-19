from torch import Tensor
import torch.nn as nn

from transformer.layers import (
    LayerNormalization,
    MultiHeadAttention,
    PositionWiseFeedForward,
    ResidualConnection,
)
from transformer.model.transformer_config import TransformerConfig


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.masked_attention = MultiHeadAttention(config.d_model, config.num_heads, dropout=config.attention_dropout)
        self.masked_attention_residual_connection = ResidualConnection(config.d_model, dropout=config.dropout)

        self.cross_attention = MultiHeadAttention(config.d_model, config.num_heads, dropout=config.attention_dropout)
        self.cross_attention_residual_connection = ResidualConnection(config.d_model, dropout=config.dropout)

        self.position_wise_ffn = PositionWiseFeedForward(config.d_model, config.d_ffn, dropout=config.dropout)
        self.ffn_residual_connection = ResidualConnection(config.d_model, dropout=config.dropout)

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
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = LayerNormalization(config.d_model)

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
