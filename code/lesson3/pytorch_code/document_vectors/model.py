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
    def __init__(self, D_in_words, D_in_documents, embedding_dim,
        num_hidden_units, window_size, embeddings):
        """
        Initialize weights.
        """
        super(MLP, self).__init__()

        # Embeddings
        self.embeddings = embeddings
        self.word_embeddings = nn.Embedding(
            num_embeddings=D_in_words,
            embedding_dim=embedding_dim,
            )
        self.doc_embeddings = nn.Embedding(
            num_embeddings=D_in_documents,
            embedding_dim=embedding_dim,
            )

        self.fc1 = nn.Linear(
            embedding_dim*2, num_hidden_units)
        self.fc2 = nn.Linear(num_hidden_units, D_in_words)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights.
        """
        self.word_embeddings.weight.data = self.embeddings
        self.word_embeddings.weight.requires_grad=False

        self.doc_embeddings.weight.data.uniform_(-np.sqrt(3), np.sqrt(3))

    def forward(self, doc_inputs, word_inputs):
        """
        Forward pass.
        """

        # Representations
        doc_representation = self.doc_embeddings(doc_inputs)
        words_representation = \
            torch.sum(self.word_embeddings(word_inputs), dim=1).squeeze()

        x = torch.cat([words_representation, doc_representation], 1)
        z = self.fc1(x)
        z = self.fc2(z)
        return z

