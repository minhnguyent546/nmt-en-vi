from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class TransformerConfig:
    src_vocab_size: int
    target_vocab_size: int
    src_seq_length: int
    target_seq_length: int
    src_pad_token_id: int
    target_pad_token_id: int
    device: torch.device | Literal['auto'] = 'auto'
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ffn: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
