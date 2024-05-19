import math

from torch import Tensor
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size (int): size of the vocabulary
            d_model (int): dimension of the embedding vectors
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input Tensor, shape ``(batch_size, seq_length)``

        Returns:
            embedding vectors, shape ``(batch_size, seq_length, d_model)``
        """
        coeff = math.sqrt(self.d_model)
        return self.embedding(x) * coeff
