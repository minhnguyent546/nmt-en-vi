import random
from typing import Generator
import numpy as np

import torch
from torch import Tensor

from tokenizers import Tokenizer

from nmt.constants import SpecialToken

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

def combined_iterator(*iterators) -> Generator:
    for values in zip(*iterators):
        for value in values:
            yield value

def remove_end_tokens(
    tokens: Tensor | np.ndarray,
    tokenizer: Tokenizer,
    *,
    contains_id: bool = False
) -> Tensor | np.ndarray:
    assert tokens.ndim == 1
    if isinstance(tokens, Tensor) and tokens.numel() == 0:
        return tokens
    if isinstance(tokens, np.ndarray) and tokens.size == 0:
        return tokens

    first_token = tokenizer.id_to_token(tokens[0]) if contains_id else tokens[0]
    last_token = tokenizer.id_to_token(tokens[-1]) if contains_id else tokens[-1]
    if first_token == SpecialToken.SOS:
        tokens = tokens[1:]
    if last_token == SpecialToken.EOS:
        tokens = tokens[:-1]
    return tokens

def tokens_to_text(tokens: list[str]) -> str:
    text = ' '.join(tokens)

