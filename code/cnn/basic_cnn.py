from __future__ import (
    print_function,
)

"""
basics.py
Example of NN in numpy/PyTorch.
Note: Strong overfitting but example is
      just to show forward/backward pass.
"""

__author__ = "Goku Mohandas"
__email__ = "gokumd@gmail.com"

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchvision import (
    datasets,
    transforms,
)

from torch.autograd import (
    Variable,
)

from torchvision import (
    datasets as dset,
    transforms,
)

basedir = os.path.abspath(os.path.dirname(__file__))

def get_args():
    """
    Arguments
    """
    parser = argparse.ArgumentParser (description='Main arguments')
    parser.add_argument ('--seed', type=int, default=1234)
    parser.add_argument ('--num_workers', type=int, default=4) # load data
    parser.add_argument ('--num_epochs', type=int, default=10)
    parser.add_argument ('--batch_size', type=int, default=64)
    parser.add_argument ('--test_batch_size', type=int, default=1000)
    parser.add_argument ('--learning_rate', type=float, default=0.01)
    parser.add_argument ('--log_interval', type=int, default=10, help='batches')
    parser.add_argument ('--no_cuda', action='store_true', default=False)

    return parser.parse_args ()

def load_data(batch_size, num_workers):
    """
    Load data from torchvision.
    """



def main(FLAGS):
    """
    """

    # Load data loaders
    load_data(FLAGS.batch_size, FLAGS.num_workers)


if __name__ == '__main__':

    FLAGS = get_args()

    # CUDA
    FLAGS.cuda = not FLAGS.no_cuda and torch.cuda.is_available()

    # Seeding
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed_all(FLAGS.seed)

    main(FLAGS)









