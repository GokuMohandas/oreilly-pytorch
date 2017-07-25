"""
infer.py
"""
from __future__ import (
    print_function,
)

import os
import csv
import re
import json
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import (
    Variable,
)

from config import (
    basedir,
)

from data import (
    normalize_string,
    convert_sentence,
)

from utils import (
    pad,
)

def infer(data_dir, model_name, sentence=None):
    """
    """

    # Load components
    with open(os.path.join(basedir, data_dir, 'char2index.json'), 'r') as f:
        char2index = json.load(f)
    with open(os.path.join(basedir, data_dir, 'index2class.json'), 'r') as f:
        index2class = json.load(f)

    # Enter the sentence
    print ("Classes:", index2class.values())
    if not sentence:
        sentence = input("Please enter the sentence: ")

    # Normalize the sentece
    sentence = normalize_string(sentence)

    # Convert sentence(s) to indexes
    input_ = convert_sentence(
        sentence=sentence,
        char2index=char2index,
        )

    # Convert to model input
    input_, _ = pad(
        inputs=[input_],
        char2index=char2index,
        max_length=len(input_),
        )

    # Convert to Variable
    X = Variable(torch.FloatTensor(input_), requires_grad=False)

    # Load the model
    model = torch.load(os.path.join(
        basedir, "data", data_dir.split("/")[-1], model_name))

    # Feed through model
    model.eval()
    scores = model(X)
    probabilities = F.softmax(scores)

    # Sorted probabilities
    sorted_, indices = torch.sort(probabilities, descending=True)
    for index in indices[0]:
        print ("%s - %i%%" % (
            index2class[str(index.data[0])],
            100.0*probabilities.data[0][index.data[0]]))
