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
    parser.add_argument ('--num_classes', type=int, default=5)
    parser.add_argument ('--num_hidden_units', type=int, default=100)
    parser.add_argument ('--regularization', type=float, default=1e-3)
    parser.add_argument ('--learning_rate', type=float, default=1.0)
    parser.add_argument ('--num_epochs', type=int, default=5000)

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
    y = np.zeros (N*C, dtype='int_')
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

def plot_model(X, y, w1, b1, w2, b2):
    """
    Plot the model.
    """
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], w1) + b1), w2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def numpy_version(X, y, num_hidden_units, num_classes, regularization,
    learning_rate, num_epochs):
    """
    Implement NN with numpy.
    """

    # Dimensions
    N = len(X) # num. samples
    D_in = len(X[0]) # input dim.
    H = num_hidden_units # hidden dim.
    D_out = num_classes # output dim.

    # Weights
    w1 = 0.01 * np.random.randn(D_in, H)
    w2 = 0.01 * np.random.randn(H, D_out)

    b1 = np.zeros((1, H))
    b2 = np.zeros((1, D_out))

    for epoch in range(num_epochs):

        # Forward pass
        h = np.dot(X, w1) + b1
        h_relu = np.maximum(0, h)
        scores = np.dot(h_relu, w2) + b2
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N, D_out]

        # Cross entropy loss
        y_true_logprobs = -np.log(probs[range(N), y])
        loss = np.sum(y_true_logprobs) / N
        loss += 0.5*regularization*np.sum(w1*w1) + 0.5*regularization*np.sum(w2*w2)

        # Backpropagation
        dJ__dscores = probs
        dJ__dscores[range(N), y] -= 1
        dJ__dscores /= N

        dJ__dw2 = np.dot(h_relu.T, dJ__dscores)
        dJ__db2 = np.sum(dJ__dscores, axis=0, keepdims=True)
        dJ__dh_relu = np.dot(dJ__dscores, w2.T)
        dJ__dh_relu[h_relu <= 0] = 0 # dJ__dh
        dJ__dh = dJ__dh_relu
        dJ__dw1 = np.dot(X.T, dJ__dh)
        dJ__db1 = np.sum(dJ__dh, axis=0, keepdims=True)

        # Derivative of regularization component
        dJ__dw2 += regularization * w2
        dJ__dw1 += regularization * w1

        # Gradient descent
        w1 -= learning_rate * dJ__dw1
        b1 -= learning_rate * dJ__db1
        w2 -= learning_rate * dJ__dw2
        b2 += -learning_rate * dJ__db2

        # Verbose
        if (epoch % 1000 == 0) or (epoch == num_epochs-1):

            # Accuracy
            y_pred = np.argmax(scores, axis=1)
            train_accuracy = (np.mean(y_pred == y))

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f" %
                (epoch, loss, train_accuracy))

    # Plot trained model
    plot_model(X, y, w1, b1, w2, b2)



def pytorch_tensors(X, y, num_hidden_units, num_classes, regularization,
    learning_rate, num_epochs, dtype=torch.FloatTensor):
    """
    Implement NN with PyTorch tensors.
    """

    # Dimensions
    N = len(X) # num. samples
    D_in = len(X[0]) # input dim.
    H = num_hidden_units # hidden dim.
    D_out = num_classes # output dim.

    # Convert data to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    # Weights
    w1 = torch.randn(D_in, H).type(dtype)
    w2 = torch.randn(H, D_out).type(dtype)

    b1 = torch.zeros((N, H))
    b2 = torch.zeros((N, D_out))

    for epoch in range(num_epochs):

        # Forward pass
        h = X.mm(w1) + b1
        h_relu = h.clamp(min=0)
        scores = h_relu.mm(w2) + b2
        exp_scores = torch.exp(scores)
        probs = exp_scores / torch.sum(exp_scores, dim=1).repeat(1,num_classes) # [N, D_out]

        # Cross entropy loss
        y_true_logprobs = -torch.log(torch.gather(probs, 1, y.unsqueeze(1)))
        loss = torch.sum(y_true_logprobs) / N
        loss += 0.5*regularization*torch.sum(w1*w1) + 0.5*regularization*torch.sum(w2*w2)

        print (shit)

        # Backpropagation
        dJ__dscores = probs
        dJ__dscores[range(N), y] -= 1
        dJ__dscores /= N

        dJ__dw2 = np.dot(h_relu.T, dJ__dscores)
        dJ__db2 = np.sum(dJ__dscores, axis=0, keepdims=True)
        dJ__dh_relu = np.dot(dJ__dscores, w2.T)
        dJ__dh_relu[h_relu <= 0] = 0 # dJ__dh
        dJ__dh = dJ__dh_relu
        dJ__dw1 = np.dot(X.T, dJ__dh)
        dJ__db1 = np.sum(dJ__dh, axis=0, keepdims=True)

        # Derivative of regularization component
        dJ__dw2 += regularization * w2
        dJ__dw1 += regularization * w1

        # Gradient descent
        w1 -= learning_rate * dJ__dw1
        b1 -= learning_rate * dJ__db1
        w2 -= learning_rate * dJ__dw2
        b2 += -learning_rate * dJ__db2

        # Verbose
        if (epoch % 1000 == 0) or (epoch == num_epochs-1):

            # Accuracy
            y_pred = np.argmax(scores, axis=1)
            train_accuracy = (np.mean(y_pred == y))

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f" %
                (epoch, loss, train_accuracy))

    # Plot trained model
    plot_model(X, y, w1, b1, w2, b2)

if __name__ == '__main__':

    FLAGS = get_args ()
    X, y = get_data (FLAGS.seed, FLAGS.num_samples, FLAGS.dimensions,
        FLAGS.num_classes)
    #plot_data(X, y)

    # Numpy
    #numpy_version(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
    #    FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs)

    pytorch_tensors(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
        FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs)






