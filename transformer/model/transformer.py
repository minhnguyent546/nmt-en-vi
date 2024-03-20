import torch.nn as nn
from torch import Tensor

from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder
from ..embedding import Embeddings, PositionalEncoding
from ..utils import functional as fun

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
    ):
        """
        Args:
            encoder (Encoder): encoder model
            decoder (Decoder): decoder model
            src_embed (Embeddings): source embedding
            target_embed (Embeddings): target embedding
            src_pe (PositionalEncoding): source positional encoding
            target_pe (PositionalEncoding): target positional encoding
            last_linear (nn.Linear): the last linear layer
            src_pad_token_id (int): id of the padding token for source
            target_pad_token_id (int): id of the padding token for target
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

    def linear(self, x: Tensor) -> Tensor:
        return self.last_linear(x)

def make_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_length: int,
    target_seq_length: int,
    src_pad_token_id: int,
    target_pad_token_id: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ffn: int = 2048,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
) -> Transformer:
    src_embed = Embeddings(src_vocab_size, d_model)
    target_embed = Embeddings(target_vocab_size, d_model)

    src_pe = PositionalEncoding(d_model, src_seq_length, dropout_rate=dropout_rate)
    target_pe = PositionalEncoding(d_model, target_seq_length, dropout_rate=dropout_rate)

    encoder_layer = EncoderLayer(d_model,
                                 num_heads,
                                 d_ffn,
                                 dropout_rate=dropout_rate,
                                 attention_dropout_rate=attention_dropout_rate)
    decoder_layer = DecoderLayer(d_model,
                                 num_heads,
                                 d_ffn,
                                 dropout_rate=dropout_rate,
                                 attention_dropout_rate=attention_dropout_rate)
    encoder = Encoder(encoder_layer, d_model, num_layers)
    decoder = Decoder(decoder_layer, d_model, num_layers)
    last_linear = nn.Linear(d_model, target_vocab_size)

    # create a transformer
    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        target_embed,
        src_pe,
        target_pe,
        last_linear,
        src_pad_token_id,
        target_pad_token_id
    )

    print(f'Model has {fun.count_parameters(transformer)} learnable parameters', )

    # initialize the parameters with Xavier/Glorot
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer
