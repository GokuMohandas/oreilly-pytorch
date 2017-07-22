"""
infer.py
Get similar words for a
given word using embeddings.
"""

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
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import (
    TSNE,
)

from config import (
    basedir,
)

def infer(FLAGS):
    """
    Inference.
    """

    # Retrieve embeddings for docs
    words = ["tennis", "wimbledon", "icecream", "cake", "bear", "pie"]

     # Get index in doc embeddings
    with open(os.path.join(basedir, FLAGS.data_dir, "doc_to_idx.json"), 'r') as f:
        doc_to_idx = json.load(f)

    # Load the trained model
    model = torch.load(os.path.join(basedir, FLAGS.data_dir, "model.pt"))
    doc_embeddings = model.doc_embeddings.weight.data

    my_embeddings = np.array(
        [doc_embeddings[doc_to_idx[word]].numpy() for word in words])

    # Use TSNE model to reduce dimensionality
    model = TSNE(n_components=2, random_state=0)
    points = model.fit_transform(my_embeddings)

    # Visualize
    for i, word in enumerate(words):
        x, y = points[i, 0]*1e4, points[i, 1]*1e4
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(25, 5),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()





