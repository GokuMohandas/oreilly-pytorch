"""
Utility functions
"""
from __future__ import (
    print_function,
)

import os
import csv
import re
import json
import unicodedata
import random
import numpy as np
import pandas as pd

import torch.utils.data

from tqdm import (
    tqdm,
)

from random import (
    shuffle,
)

from config import (
    basedir,
    get_config,
)

def sample(data, data_dir):
    """
    Sample a data point.
    """

    # Choose a random sample
    rand_index = random.randint(0, len(data)-1)
    entry = data[rand_index]

    print ("==> Sample:")
    print (np.shape(entry[0]))
    print ("Processed input:\n", entry[0])
    print ("Processed output:", entry[1])

def generate_epoch(entries, num_epochs, batch_size):
    """
    Generate epochs.
    """

    # Separate into inputs
    inputs = []
    targets = []
    for entry in entries:
        inputs.append(entry[0])
        targets.append(entry[1])

    for epoch_num in range(num_epochs):
        yield generate_batch(inputs, targets, batch_size)

def generate_batch(inputs, targets, batch_size):
    """
    Generate batches.
    """
    data_size = len(inputs)
    num_batches = data_size//batch_size

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, data_size)

        yield inputs[start_index:end_index], \
              targets[start_index:end_index], \
              len(targets[start_index:end_index])

def pad(inputs, char2index, max_length=None):
    """
    Pad the inputs.
    inputs [[3, 34, 12], [4, 12], ...]
    """

    # If not max_len, assume first input
    # is largest (inputs already sorted by size)
    if max_length is None:
        max_length = len(inputs[0])

    # Pad the inputs
    lengths = []
    padded_inputs = []
    for item in inputs:
        lengths.append(len(item))

        # Create padding
        padding = np.zeros((max_length-len(item), len(char2index)))
        for i, row in enumerate(padding):
            padding[i][char2index["__PAD__"]] = 1.

        # Add padding to current input
        padded_input = np.vstack((item, padding))

        # Add to padded inputs
        padded_inputs.append(padded_input)

    return padded_inputs, lengths
