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

from config import (
    basedir,
)

def infer(FLAGS):
    """
    Inference.
    """

    # Ask for a word
    word = input("Enter a word: ").lower()

    # Get index in embeddings
    with open(os.path.join(basedir, FLAGS.data_dir, "word_to_idx.json"), 'r') as f:
        word_to_idx = json.load(f)
    with open(os.path.join(basedir, FLAGS.data_dir, "idx_to_word.json"), 'r') as f:
        idx_to_word = json.load(f)
    index = word_to_idx[word] if word in word_to_idx else word_to_idx["UNK"]

    # Unknown word
    if word not in word_to_idx:
        print ("%s is not part of our corpus." % word)
        return

    # Load the trained model
    model = torch.load(os.path.join(basedir, FLAGS.data_dir, "model.pt"))
    embeddings = model.embedding.weight

    # Cosine similarity
    normalized_embeddings = embeddings/\
    ((embeddings**2).sum(0)**0.5).expand_as(embeddings)
    normalized_embedding_for_word = normalized_embeddings[index]

    # similarity score
    similarity, words = torch.topk(torch.mv(normalized_embeddings,
        normalized_embedding_for_word),FLAGS.TOP_K+1)

    print ("Close to", word, ":", [
        idx_to_word[str(word.data[0])] for word in words[1:]])






