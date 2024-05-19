import math

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as Fun

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): dimension of the embedding vectors
            num_heads (int): number of attention heads
            dropout (float): dropout rate in attention
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
        self.dropout = nn.Dropout(dropout)

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
            mask (Tensor): mask tensor, shape ``(batch_size, 1, 1, k_length)`` (default: None)

        Returns:
            x (Tensor): shape ``(batch_size, q_length, d_model)``
        """
        batch_size = query.size(0)

        # find q, k, v tensors
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # split q, k, v into multiple heads
        # q: (batch_size, num_heads, q_length, d_k)
        # k: (batch_size, num_heads, k_length, d_k)
        # v: (batch_size, num_heads, v_length, d_v)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # x: (batch_size, num_heads, q_length, d_v)
        # attention_probs: (batch_size, num_heads, q_length, k_length)
        x, attention_probs = scaled_dot_product(q, k, v, mask=mask, dropout=self.dropout)
        self.attention_probs = attention_probs

        # (batch_size, q_length, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.w_o(x)
        return x

def scaled_dot_product(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | float | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Args:
        query (Tensor): query tensor, shape ``(batch_size, num_heads, q_length, d_k)``
        key (Tensor): key tensor, shape ``(batch_size, num_heads, k_length, d_k)``
        value (Tensor): value tensor, shape ``(batch_size, num_heads, v_length, d_v)``
        mask (Tensor): mask tensor, shape ``(batch_size, 1, 1, k_length)`` (default: None)
        dropout (nn.Dropout): dropout layer (default: None)
    Returns:
        values (Tensor): attention tensor, shape ``(batch_size, num_heads, q_length, d_v)``
        attention_probs (Tensor): softmax score, shape ``(batch_size, num_heads, q_length, k_length)``
    """

    d_k = q.size(-1)
    # attention_probs: (batch_size, num_heads, q_length, k_length)
    attention_probs = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_probs.masked_fill_(mask == False, float('-inf'))

    attention_probs = Fun.softmax(attention_probs, dim=-1)
    if dropout is not None:
        if isinstance(dropout, float):
            dropout = nn.Dropout(dropout)
        attention_probs = dropout(attention_probs)

    values = attention_probs @ v
    return values, attention_probs
