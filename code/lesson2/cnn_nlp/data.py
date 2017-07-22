"""
data.py
"""
from __future__ import (
    print_function,
)

import os
import csv
import re
import json
import unicodedata
import numpy as np
import pandas as pd

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

PAD_ID = 0
UNK_ID = 1

def get_classes(data_dir):
    """
    Key for classes.
    """
    classes = {}
    index = 0
    with open(os.path.join(basedir, data_dir, 'classes.txt'), 'r') as f:
        for line in f:
            classes[index] = line.strip()
            index += 1

    # Save to file (will use in other places)
    with open(os.path.join(basedir, data_dir, 'index2class.json'), 'w') as f:
        json.dump(classes, f)

def unicode_to_ascii(s):
    """
    Convert unicode string to ASCII.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    """
    Process a string:
        lower case,
        strip white space,
        remove non (a-zA-Z.!?)
    """
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_data(data_dir):
    """
    Read the data from file.
    """
    print ("\n==> Reading the data ...")

    # Combine csvs

    #train = pd.read_csv(os.path.join(basedir, data_dir, "train.csv"), header=None)
    #test = pd.read_csv(os.path.join(basedir, data_dir, "test.csv"), header=None)
    #csv_data = pd.concat([train, test], ignore_index=True)

    csv_data = pd.read_csv(os.path.join(basedir, data_dir, "test.csv"), header=None)

    # Store the processed data
    data = []

    # Process each row
    for index, row in tqdm(csv_data.iterrows()):
        _class = int(row[0])
        _description = normalize_string(row[2])

        # Store
        data.append((_description, _class))

    return data

def process_data(data, data_dir):
    """
    Process the data by
    extracting information.
    """
    print ("==> Processing the data ...")

    def add_sentence(sentence, unique_chars):
        """
        Process each word in the sentence.
        """
        for char in list(sentence):
            unique_chars.add(char)
        return unique_chars

    # One pass through data for char collection
    unique_chars = set()
    for point in data:

        # Collect chars from sentence
        unique_chars = add_sentence(
            sentence=point[0],
            unique_chars=unique_chars,
            )

    # Create dicts
    char2index = {}
    index2char = {}

    # Add PAD and UNK
    char2index["__PAD__"] = PAD_ID
    char2index["__UNK__"] = UNK_ID
    index2char[PAD_ID] = "__PAD__"
    index2char[UNK_ID] = "__UNK__"

    # Update dicts
    index = len(char2index)
    for char in unique_chars:
        char2index[char] = index
        index2char[index] = char
        index += 1

    # Save to file
    with open(os.path.join(
        basedir, "data", data_dir.split("/")[-1], "char2index.json"), 'w') as f:
        json.dump(char2index, f)
    with open(os.path.join(
        basedir, "data", data_dir.split("/")[-1], "index2char.json"), 'w') as f:
        json.dump(index2char, f)

    return char2index, index2char

def convert_sentence(sentence, char2index):
    """
    Convert a sentence to indexes.
    """
    X = np.zeros((len(list(sentence)), len(char2index)))
    for i, char in enumerate(list(sentence)):
        if char in char2index:
            index = char2index[char]
        else:
            index = char2index["UNK"]
        X[i][index] = 1.

    return X

def convert_data(data, char2index):
    """
    Create input and output arrays.
    """
    print ("==> Converting the data ...")

    # Another pass to process data (could do it all in one pass but
    # separating for clarity.)
    processed_data = []
    for item in tqdm(data):
        X = convert_sentence(
            sentence=item[0],
            char2index=char2index,
            )
        y = item[1]-1 # start at 0

        processed_data.append((X, y))

    return processed_data

def split_data(data, split_ratio):
    """
    Split data into
    train/test sets.
    """
    print ("==> Splitting the data ...")

    # Shuffle the data (in-place)
    shuffle(data)

    # Split data
    train_end_index = int(split_ratio*len(data))
    train_data = data[:train_end_index]
    test_data = data[train_end_index:]

    # Sort the datasets independently to make it
    # computationally efficient for padding/batch processing.
    # Sort in descending order so large enough buffers will be
    # prepared in the beginning :)
    train_data = sorted(train_data,
        key=lambda item: len(item[0]),
        reverse=True,
        )
    test_data = sorted(test_data,
        key=lambda item: len(item[0]),
        reverse=True,
        )

    return train_data, test_data

def main(data_dir, split_ratio):
    """
    """

    # Key for classes
    get_classes(
        data_dir=data_dir,
        )

    # Read the data
    data = read_data(
        data_dir=data_dir,
        )

    # Process the data
    char2index, index2char = process_data(
        data=data,
        data_dir=data_dir,
        )

    # Convert the data to arrays
    converted_data = convert_data(
        data=data,
        char2index=char2index,
        )

    train_data, test_data = split_data(
        data=converted_data,
        split_ratio=split_ratio,
        )

    print ("%i training samples and %i test samples." %
        (len(train_data), len(test_data)))

    return train_data, test_data
