"""
model.py
Embeddings model.
"""

from __future__ import (
    print_function,
)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import (
    Variable,
)
from torch.nn import (
    init,
)

class MLP(nn.Module):
    """
    """
    def __init__(self, D_in, H):
        """
        Initialize weights.
        """
        super(MLP, self).__init__()

         # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=D_in,
            embedding_dim=H,
            )
        self.fc2 = nn.Linear(H, D_in)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights.
        """
        self.embedding.weight.data.uniform_(-np.sqrt(3), np.sqrt(3))

    def forward(self, x):
        """
        Forward pass.
        """
        z = self.embedding(x)
        z = self.fc2(z)
        return z

