import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as Fun
import math
import utils as Util

class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size (int): size of the vocabulary
            d_model (int): dimension of the embedding vectors
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input Tensor, shape ``(batch_size, seq_length)``

        Returns:
            embedding vectors, shape ``(batch_size, seq_length, d_model)``
        """
        coeff = math.sqrt(self.d_model)
        return self.embedding(x) * coeff

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
        positions = torch.arange(max_seq_length, dtype=torch.float).unsqueeze(1) # (max_seq_length, 1)
        div_term = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float) * math.log(10000.0) / d_model)

        # calculate sine for even indices
        pe[:, ::2] = torch.sin(positions * div_term)

        # calculate cos for odd indices
        pe[:, 1::2] = torch.cos(positions * div_term)
        
        # add a dimension to compatible with batch dimension
        pe = pe.unsqueeze(0) # (1, max_seq_length, d_model)

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

def scaled_dot_product(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """
    Args:
        query (Tensor): query tensor, shape ``(batch_size, num_heads, q_length, d_model)``
        key (Tensor): key tensor, shape ``(batch_size, num_heads, k_length d_model)``
        value (Tensor): value tensor, shape ``(batch_size, num_heads, v_length, d_model)``
        mask (Tensor | None): mask for decoder

    Returns:
        values (Tensor): attention tensor, shape ``(batch_size, num_heads, q_length, d_model)``
        attention_probs (Tensor): softmax score, shape ``(batch_size, num_heads, q_length, k_length)``
    """
    d_k = q.size(-1)
    attention_probs = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_probs.masked_fill_(mask == False, -1e9)
    attention_probs = Fun.softmax(attention_probs, dim=-1)
    values = attention_probs @ v
    return values, attention_probs

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            num_heads (int): number of attention heads
            dropout_rate (float): dropout rate
        """ 
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, 'd_model should be divisible by num_heads'
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        """
        Args:
            query (Tensor): query tensor, shape ``(batch_size, q_length, d_model)``
            key (Tensor): key tensor, shape ``(batch_size, k_length d_model)``
            value (Tensor): value tensor, shape ``(batch_size, v_length, d_model)``
            mask (Tensor | None): mask for decoder
        
        Returns:
            x (Tensor): shape ``(batch_size, q_length, d_model)``
        """
        batch_size = query.size(0)

        # find q, k, v tensors
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # split q, k, v into multiple heads
        # q: (batch_size, num_heads, q_length, d_key)
        # k: (batch_size, num_heads, k_length, d_key)
        # v: (batch_size, num_heads, v_length, d_key)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # x: (batch_size, num_heads, q_length, d_key)
        # attention_probs: (batch_size, num_heads, q_length, k_length)
        x, attention_probs = scaled_dot_product(q, k, v, mask=mask) 
        self.attention_probs = attention_probs

        # (batch_size, q_length, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.w_o(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, features, eps: float = 1e-7):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor, shape ``(batch_size, seq_length, d_model)``
        Returns:
            y (Tensor): standardized tensor, shape ``(batch_size, seq_length, d_model)``
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt() # prevent std become too large when var is ~ zero
        y = (x - mean) / std
        y = self.gamma * y + self.beta
        return y

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout_rate: float = 0.1):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            d_ffn (int): dimension of the hidden layer
            dropout_rate (float): dropout rate
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn)
        self.linear_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
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
        return x + self.dropout(sublayer(self.norm(x)))
        # or maybe: self.norm(x + self.dropout(sublayer(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout_rate: float = 0.1):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            num_heads (int): number of attention heads
            d_ffn (int): dimension of feed-forward network
            dropout_rate (float): dropout rate
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate=dropout_rate)
        self.attention_residual_connection = ResidualConnection(d_model, dropout_rate=dropout_rate)

        self.position_wise_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout_rate=dropout_rate)
        self.ffn_residual_connection = ResidualConnection(d_model, dropout_rate=dropout_rate)

    def forward(self, inputs: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            inputs (Tensor): positionally embedded inputs tensor, shape ``(batch_size, seq_length, d_model)``
            src_mask (Tensor | None): mask tensor, shape ``(batch_size, seq_length, seq_length)``

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
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ffn: int,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            num_layers (int): number of encoder layers
            num_heads (int): number of attention heads
            d_ffnn (int): dimension of feed-forward network
            dropout (float): dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ffn, dropout_rate=dropout_rate)
            for layer in range(num_layers)]
        )
        self.norm = LayerNormalization(d_model)

    def forward(self, inputs: Tensor, src_mask: Tensor | None = None):
        """
        Args:
            inputs (Tensor): positionally embedded inputs tensor, shape ``(batch_size, seq_length, d_model)``
            src_mask (Tensor): mask tensor, shape ``(batch_size, sequence_length, sequence_length)``

        Returns:
            output (Tensor): sequences after self-attention ``(batch_size, sequence_length, d_model)``
        """
        for layer in self.layers:
            inputs = layer(inputs, src_mask=src_mask)

        inputs = self.norm(inputs)
        return inputs

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout_rate: float = 0.1):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            num_heads (int): number of attention heads
            d_ffn (int): dimension of feed-forward network
            dropout (float): dropout rate
        """
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout_rate=dropout_rate)
        self.masked_attention_residual_connection = ResidualConnection(d_model, dropout_rate=dropout_rate)

        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_rate=dropout_rate)
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
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ffn: int,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            num_layers (int): number of decoder layers
            num_heads (int): number of attention heads
            d_ffn (int): dimension of feed-forward network
            dropout (float): dropout rate
        """  

        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(
                d_model,
                num_heads,
                d_ffn,
                dropout_rate=dropout_rate
            ) for layer in range(num_layers)]
        )
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

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model (int): dimension of embedding vectors
            vocab_size (int): size of the vocabulary
        """

        super().__init__()
        self.projector = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor, shape ``(batch_size, seq_length, d_model)``

        Returns:
            output (Tensor): shape ``(batch_size, seq_length, vocab_size)``
        """
        x = self.projector(x)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embeddings,
        target_embed: Embeddings,
        src_pe: PositionalEncoding,
        target_pe: PositionalEncoding,
        projector: ProjectionLayer
    ):
        """
        Args:
            encoder (Encoder): encoder model
            decoder (Decoder): decoder model
            src_embed (Embeddings): source embedding
            target_embed (Embeddings): target embedding
            src_pe (PositionalEncoding): source positional encoding
            target_pe (PositionalEncoding): target positional encoding
            projector (ProjectionLayer): projection layer
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pe = src_pe
        self.target_pe = target_pe
        self.projector = projector

    def encode(self, src: Tensor, src_mask: Tensor):
        src = self.src_embed(src)
        src = self.src_pe(src)
        src = self.encoder(src, src_mask=src_mask)
        return src

    def decode(
        self, 
        src: Tensor,
        target: Tensor,
        src_mask: Tensor,
        target_mask: Tensor
    ) -> Tensor:
        target = self.target_embed(target)
        target = self.target_pe(target)
        target = self.decoder(src, target, src_mask, target_mask)
        return target

    def project(self, x):
        x = self.projector(x)
        return x

def make_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_length: int,
    target_seq_length: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ffn: int = 2048,
    dropout_rate: float = 0.1,
) -> Transformer:
    src_embed = Embeddings(src_vocab_size, d_model)
    target_embed = Embeddings(target_vocab_size, d_model)

    src_pe = PositionalEncoding(d_model, src_seq_length, dropout_rate=dropout_rate)
    target_pe = PositionalEncoding(d_model, target_seq_length, dropout_rate=dropout_rate)

    encoder = Encoder(d_model, num_layers, num_heads, d_ffn, dropout_rate=dropout_rate)
    decoder = Decoder(d_model, num_layers, num_heads, d_ffn, dropout_rate=dropout_rate)

    projector = ProjectionLayer(d_model, target_vocab_size)

    # create a transformer
    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        target_embed,
        src_pe,
        target_pe,
        projector
    )

    print('number of parameters:', Util.count_parameters(transformer))

    # initialize the parameters with Xavier/Glorot
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer 
