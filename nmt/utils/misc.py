import random
import numpy as np
import torch

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
