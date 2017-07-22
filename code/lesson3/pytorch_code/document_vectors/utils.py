"""
utils.py
Utility functions.
"""

from __future__ import (
    print_function,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import (
    Variable,
)

import os
import json
import time
import numpy as np

from tqdm import (
    tqdm,
)

from pycrayon import (
    CrayonClient,
)

from config import (
    basedir,
)

def load_embeddings(embedding_dir, embedding_dim):
    """
    Load embeddings from
    text file.
    """

    # Read from file
    glove_file = os.path.join(
            embedding_dir, 'glove.6B.%id.txt'%embedding_dim)
    with open(glove_file, 'r') as f:
        lines = f.readlines()

    # Load embeddings
    word_to_idx = {}
    embeddings = []
    for i, line in enumerate(tqdm(lines)):
        split_line = line.split(' ')

        word = split_line[0]
        embedding = [float(val) for val in split_line[1:]]

        word_to_idx[word] = i
        embeddings.append(torch.FloatTensor(embedding))

    return word_to_idx, embeddings

def get_embeddings(embedding_dir, embedding_dim, words):
    """
    Create embeddings for a specific
    set of words.
    """
    word_to_idx, embeddings = load_embeddings(
        embedding_dir=embedding_dir,
        embedding_dim=embedding_dim,
        )

    my_embeddings = torch.zeros((len(words), embedding_dim))
    for i, word in enumerate(words):
        if word in word_to_idx:
            my_embeddings[i,:] = embeddings[word_to_idx[word]]
        else:
            my_embeddings[i,:] = torch.zeros((embedding_dim))

    return my_embeddings

def generate_epoch(data_loader, num_epochs):
    """
    Generate epochs.
    """

    for epoch_num in range(num_epochs):
        yield data_loader
