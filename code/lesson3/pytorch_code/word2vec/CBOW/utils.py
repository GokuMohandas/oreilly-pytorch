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

def generate_epoch(data_loader, num_epochs):
    """
    Generate epochs.
    """

    for epoch_num in range(num_epochs):
        yield data_loader
