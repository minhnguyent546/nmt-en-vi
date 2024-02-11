import torch

def count_parameters(model):
    return sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

def create_mask(seq_length: int):
    tril_mask = torch.tril(torch.ones(1, seq_length, seq_length), diagonal=1).int() # (1, seq_length, seq_length)
    return tril_mask
