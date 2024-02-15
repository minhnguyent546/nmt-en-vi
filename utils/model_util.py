import torch
from torch import Tensor

from pathlib import Path

def count_parameters(model) -> int:
    return sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

def create_mask(seq_length: int) -> Tensor:
    tril_mask = torch.tril(torch.ones(1, seq_length, seq_length)).bool() # (1, seq_length, seq_length)
    return tril_mask

def get_weights_file_path(epoch: str, config: dict) -> str:
    model_dir = config['model_dir']
    model_basename = config['model_basename']
    model_file = f"{model_basename}_{epoch}.pt"
    return str(Path(model_dir) / model_file)

def get_latest_weights_file_path(config: dict) -> str | None:
    model_dir = config['model_dir']
    model_basename = config['model_basename']
    saved_files = list(Path(model_dir).glob(f'{model_basename}_*.pt'))
    if len(saved_files) > 0:
        latest_file = sorted(saved_files)[-1]
        return str(latest_file)
    return None
