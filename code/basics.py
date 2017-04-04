from __future__ import (
    print_function,
)

"""
basics.py
"""

__author__ = "Goku Mohandas"
__email__ = "gokumd@gmail.com"

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import (
    datasets,
    transforms,
)

from torch.autograd import (
    Variable,
)

def get_args():
    """
    Arguments
    """
    parser = argparse.ArgumentParser (description='Main arguments')
    parser.add_argument ('--seed', type=int, default=1234)
    parser.add_argument ('--num_samples', type=int, default=200)
    parser.add_argument ('--dimensions', type=int, default=2)
    parser.add_argument ('--num_classes', type=int, default=4)
    return parser.parse_args ()

def get_data(seed, num_samples, dimensions, num_classes):
    """
    Create input data.
    """
    np.random.seed (seed)
    N = num_samples
    D = dimensions
    C = num_classes

    # Create spiral data
    X = np.zeros ((N*C, D))
    y = np.zeros (N*C)
    for j in xrange (C):
        ix = range (N*j,N*(j+1))
        r = np.linspace (0.0,1,N)
        t = np.linspace (j*4,(j+1)*4,N) + np.random.randn (N)*0.1
        X[ix] = np.c_[r*np.sin (t), r*np.cos (t)]
        y[ix] = j

    print ("X:", (np.shape (X)))
    print ("y:", (np.shape (y)))

    return X, y

def plot_data(X, y):
    """
    Plot the data.
    """
    plt.scatter (X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.show ()

def numpy_version(X, y):
    """
    Implement NN with numpy.
    """
    pass



if __name__ == '__main__':

    FLAGS = get_args ()
    X, y = get_data (FLAGS.seed, FLAGS.num_samples, FLAGS.dimensions,
                     FLAGS.num_classes)
    plot_data(X, y)




