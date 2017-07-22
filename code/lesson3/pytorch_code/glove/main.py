"""
main.py
Explore GloVe embeddings.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm import (
    tqdm,
)

from sklearn.manifold import (
    TSNE,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import (
    Variable,
)
from torch.nn import (
    init,
)

# Logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

# Arguments
basedir = os.path.abspath(os.path.dirname(__file__))
embedding_dir = os.path.join(basedir, "../../../../embeddings/glove")
embedding_dim = 100

def pretty_print(results):
    """
    Pretty print embedding results.
    """
    for item in results:
        print ("[%.2f] - %s"%(item[1], item[0]))

def get_embedding(word, word_to_idx, embeddings):
    """
    """
    if word.lower() in word_to_idx:
        word_embedding = embeddings[word_to_idx[word.lower()]]
    else:
        word_embedding = torch.zeros((embedding_dim))

    return word_embedding

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

def get_closest(word_embedding, word_to_idx, embeddings, n=5):
    """
    Get the n closest
    words to your word.
    """

    # Calculate distances to all other words
    distances = [(
        w, torch.dist(
            word_embedding, embeddings[word_to_idx[w]])) for w in word_to_idx]
    return sorted(distances, key=lambda x: x[1])[1:n+1]

def get_analogy(word1, word2, word3, word_to_idx, embeddings):
    """
    Get the word to
    complete an analogy.
    """

    # Find closest word to analogy answer
    print('[%s : %s :: %s : ?]' % (word1, word2, word3))
    e1 = get_embedding(word1, word_to_idx, embeddings)
    e2 = get_embedding(word2, word_to_idx, embeddings)
    e3 = get_embedding(word3, word_to_idx, embeddings)

    word_embedding = e2 - e1 + e3
    closest_words = get_closest(word_embedding, word_to_idx, embeddings, n=8)

    # Filter the results
    results = []
    for result in closest_words:
        if result[0] not in (word1, word2, word3):
            results.append(result)
    return results[:5]

def visualize_embeddings(word_to_idx, embeddings):
    """
    Visualize a sample analogy.
    """
    # Retrieve embeddings for words
    words = ["man", "king", "woman", "queen"]
    my_embeddings = np.array(
        [get_embedding(word, word_to_idx, embeddings).numpy() for word in words])

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

def main():
    """
    """

    logging.info("==> Loading Embeddings ...")
    word_to_idx, embeddings = load_embeddings(
        embedding_dir=embedding_dir,
        embedding_dim=embedding_dim,
        )

    logging.info("==> Find neighbors ...")
    word = input('Enter a word: ')
    word_embedding = get_embedding(word, word_to_idx, embeddings)
    pretty_print(get_closest(word_embedding, word_to_idx, embeddings, n=5))

    logging.info("==> Solve analogy ...")
    word1, word2, word3 = "man", "king", "woman"
    pretty_print(get_analogy(word1, word2, word3, word_to_idx, embeddings))

    logging.info("==> Sample visualization ...")
    visualize_embeddings(word_to_idx, embeddings)

if __name__ == '__main__':
    """
    """
    main()