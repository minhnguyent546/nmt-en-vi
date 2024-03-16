import copy

import torch
import torch.nn as nn
from torch import Tensor

def count_parameters(model: nn.Module) -> int:
    return sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

def create_mask(seq_length: int) -> Tensor:
    tril_mask = torch.tril(torch.ones(1, seq_length, seq_length)).bool()  # (1, seq_length, seq_length)
    return tril_mask

def get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    """Deep copy a module into stack of multiple modules"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])
