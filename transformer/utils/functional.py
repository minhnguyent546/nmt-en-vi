import copy

import torch
import torch.nn as nn
from torch import Tensor

def count_parameters(model: nn.Module) -> int:
    return sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

def create_causal_mask(seq_length: int) -> Tensor:
    """Create a causal mask for the transformer decoder

    Args:
        seq_length (int): length of the sequence

    Returns:
        Tensor: the causal mask, where the lower triangular part is filled with True, shape ``(seq_length, seq_length)``
    """
    tril_mask = torch.tril(torch.ones(seq_length, seq_length)).bool()
    return tril_mask

def create_encoder_mask(
    encoder_input: Tensor,
    pad_token_id: int,
    has_batch_dim: bool = False
) -> Tensor:
    """Create masks for encoder_input

    Args:
        encoder_input (Tensor): input tensor for the encoder, shape ``(batch_size, seq_length)`` or ``(seq_length,)``
        pad_token_id (int): id of padding token
        has_batch_dim (bool): whether the input tensor has batch dimension (default: False)

    Returns:
        Tensor: mask for encoder_input
    """

    encoder_mask = (encoder_input != pad_token_id)
    if has_batch_dim:
        encoder_mask.unsqueeze_(1).unsqueeze_(2)  # (batch_size, 1, 1, seq_length)
    else:
        encoder_mask.unsqueeze_(0).unsqueeze_(0)  # (1, 1, seq_length)

    return encoder_mask.to(encoder_input.device)

def create_decoder_mask(
    decoder_input: Tensor,
    pad_token_id: int,
    has_batch_dim: bool = False
) -> Tensor:
    """Create masks for decoder_input

    Args:
        decoder_input (Tensor): input tensor for the decoder, shape ``(batch_size, seq_length)`` or ``(seq_length,)``
        pad_token_id (int): id of padding token
        has_batch_dim (bool): whether the input tensor has batch dimension (default: False)

    Returns:
        Tensor: mask for encoder_input
    """

    causal_mask = create_causal_mask(decoder_input.size(-1)).to(decoder_input.device)
    decoder_mask = (decoder_input != pad_token_id)
    if has_batch_dim:
        decoder_mask.unsqueeze_(1).unsqueeze_(2)  # (batch_size, 1, 1, seq_length)
    else:
        decoder_mask.unsqueeze_(0).unsqueeze_(0)  # (1, 1, seq_length)

    decoder_mask = decoder_mask & causal_mask  # (batch_size, 1, seq_length, seq_length) or (1, seq_length, seq_length)
    return decoder_mask.to(decoder_input.device)

def create_masks(
    encoder_input: Tensor,
    decoder_input: Tensor,
    pad_token_id: int,
    has_batch_dim: bool = False
) -> tuple[Tensor, Tensor]:
    """Create masks for encoder_input and decoder_input

    Args:
        encoder_input (Tensor): input tensor for the encoder, shape ``(batch_size, seq_length)`` or ``(seq_length,)``
        decoder_input (Tensor): input tensor for the decoder, shape ``(batch_size, seq_length)`` or ``(seq_length,)``
        pad_token_id (int): id of padding token

    Returns:
        tuple[Tensor, Tensor]: masks for encoder_input and decoder_input
    """

    return (
        create_encoder_mask(encoder_input, pad_token_id, has_batch_dim),
        create_decoder_mask(decoder_input, pad_token_id, has_batch_dim)
    )

def get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    """Deep copy a module into stack of multiple modules"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])
