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

def get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    """Deep copy a module into stack of multiple modules"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])
