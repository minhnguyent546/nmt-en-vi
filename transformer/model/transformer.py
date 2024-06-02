from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

from transformer.embedding import Embeddings, PositionalEncoding
from transformer.model.decoder import Decoder
from transformer.model.encoder import Encoder
from transformer.model.transformer_config import TransformerConfig
from transformer.utils import functional as fun


def build_transformer(config: TransformerConfig) -> Transformer:
    src_embed = Embeddings(config.src_vocab_size, config.d_model)
    if config.shared_vocab:
        target_embed = src_embed  # using tied weights
    else:
        target_embed = Embeddings(config.target_vocab_size, config.d_model)

    src_pe = PositionalEncoding(config.d_model, config.src_seq_length, dropout=config.dropout)
    target_pe = PositionalEncoding(config.d_model, config.target_seq_length, dropout=config.dropout)

    encoder = Encoder(config)
    decoder = Decoder(config)
    last_linear = nn.Linear(config.d_model, config.target_vocab_size)
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = config.device

    # create a transformer
    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        target_embed,
        src_pe,
        target_pe,
        last_linear,
        config.src_pad_token_id,
        config.target_pad_token_id,
        device,
    )
    return transformer

class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embeddings,
        target_embed: Embeddings,
        src_pe: PositionalEncoding,
        target_pe: PositionalEncoding,
        last_linear: nn.Linear,
        src_pad_token_id: int,
        target_pad_token_id: int,
        device: torch.device,
    ):
        """
        Args:
            encoder (Encoder): encoder model
            decoder (Decoder): decoder model
            src_embed (Embeddings): source embedding
            target_embed (Embeddings): target embedding
            src_pe (PositionalEncoding): source positional encoding
            target_pe (PositionalEncoding): target positional encoding
            last_linear (nn.Linear): the last linear transformation layer
            src_pad_token_id (int): id of the padding token for source
            target_pad_token_id (int): id of the padding token for target
            device (torch.device): device type (cpu or cuda)
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pe = src_pe
        self.target_pe = target_pe
        self.last_linear = last_linear
        self.src_pad_token_id = src_pad_token_id
        self.target_pad_token_id = target_pad_token_id
        self.device = device

        self.post_init()

    def encode(self, src: Tensor, src_mask: Tensor | None = None):
        """
        Args:
            src (Tensor): source tensor, shape ``(batch_size, seq_length)``
            src_mask (Tensor): mask tensor for source, shape ``(batch_size, 1, 1, seq_length)`` (default: None)

        Returns:
            Tensor: the output tensor, shape ``(batch_size, seq_length, d_model)``
        """

        if src_mask is None:
            src_mask = fun.create_encoder_mask(src, self.src_pad_token_id, has_batch_dim=True)
        src = self.src_embed(src)
        src = self.src_pe(src)
        src = self.encoder(src, src_mask=src_mask)
        return src

    def decode(
        self,
        src: Tensor,
        target: Tensor,
        src_mask: Tensor,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            src (Tensor): encoder output, shape ``(batch_size, src_seq_length, d_model)``
            target (Tensor): target tensor, shape ``(batch_size, target_seq_length)``
            src_mask (Tensor): mask tensor for ``src``, shape ``(batch_size, 1, 1, src_seq_length)``
            target_mask (Tensor): mask tensor for ``target``, shape ``(batch_size, 1, target_seq_length, target_seq_length)`` (default: None)

        Returns:
            Tensor: the output tensor, shape ``(batch_size, seq_length, d_model)``
        """

        if target_mask is None:
            target_mask = fun.create_decoder_mask(target, self.target_pad_token_id, has_batch_dim=True)
        target = self.target_embed(target)
        target = self.target_pe(target)

        assert src.dim() == 3
        target = self.decoder(src, target, src_mask, target_mask)
        return target

    def forward(
        self,
        src: Tensor,
        target: Tensor,
        src_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            src (Tensor): source tensor, shape ``(batch_size, src_seq_length)``
            target (Tensor): target tensor, shape ``(batch_size, target_seq_length)``
            src_mask (Tensor): mask tensor for ``src``, shape ``(batch_size, 1, 1, src_seq_length)`` (default: None)
            target_mask (Tensor): mask tensor for ``target``, shape ``(batch_size, 1, target_seq_length, target_seq_length)`` (default: None)

        Returns:
            Tensor: the output tensor, shape ``(batch_size, seq_length, d_model)``
        """
        assert src.dim() == 2
        assert target.dim() == 2

        if src_mask is None:
            src_mask = fun.create_encoder_mask(src, self.src_pad_token_id, has_batch_dim=True)
        src = self.encode(src, src_mask=src_mask)
        target = self.decode(src, target, src_mask=src_mask, target_mask=target_mask)
        return target

    def linear_transform(self, x: Tensor) -> Tensor:
        return self.last_linear(x)

    def post_init(self) -> None:
        self._init_params()
        if self.src_embed is self.target_embed:
            # if we are using tied weights
            self._set_linear_transform_weights(self.src_embed)

    def _init_params(self) -> None:
        # initialize the parameters with Xavier/Glorot
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _set_linear_transform_weights(self, embeddings: Embeddings) -> None:
        self.last_linear.weight = embeddings.embedding.weight
