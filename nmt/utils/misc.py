import random
import numpy as np
import torch
from torch import Tensor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_enabled(dictionary: dict, key) -> bool:
    """
    Return true if ``key`` is in ``dictionary`` and its value is set to ``True``, ``False`` otherwise
    """
    return key in dictionary and dictionary[key] == True

def tensor_find_value(x: Tensor, value: int, kth: int | None = None) -> Tensor | None:
    """
    Return the indices of the elements that are equal to ``value`` in the tensor ``x``
    """
    matches = (x == value).nonzero()
    if matches.numel() == 0:
        return None

    if kth is not None:
        return matches[kth] if kth < matches.size(0) else None

    return matches
