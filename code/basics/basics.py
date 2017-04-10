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
    parser.add_argument ('--num_classes', type=int, default=3)
    parser.add_argument ('--num_hidden_units', type=int, default=100)
    parser.add_argument ('--regularization', type=float, default=1e-3)
    parser.add_argument ('--learning_rate', type=float, default=1.0) # overfit
    parser.add_argument ('--num_epochs', type=int, default=5000)
    parser.add_argument ('--no_cuda', action='store_true', default=False)

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
        if (epoch == 0) or (epoch % 1000 == 0) or (epoch == num_epochs-1):

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
    b1 = torch.zeros((1, H))
    b2 = torch.zeros((1, D_out))

    for epoch in range(num_epochs):

        # Forward pass
        h = X.mm(w1) + b1.repeat(N,1)
        h_relu = h.clamp(min=0)
        scores = h_relu.mm(w2) + b2.repeat(N,1)
        exp_scores = torch.exp(scores)

        probs = exp_scores / torch.sum(exp_scores, dim=1).repeat(1,num_classes) # [N, D_out]

        # Cross entropy loss
        y_true_logprobs = -torch.log(torch.gather(probs, 1, y.unsqueeze(1)))
        loss = torch.sum(y_true_logprobs) / float(N)
        loss += 0.5*regularization*torch.sum(w1*w1) + 0.5*regularization*torch.sum(w2*w2)

        # Backpropagation
        dJ__dscores = probs

        def bp_correct_class(i, row, correct_class):
            """
            1-prob for the correct class for sample i.
            """
            row[correct_class] = row[correct_class] - 1
            return row

        for i, row in enumerate(dJ__dscores):
            dJ__dscores[i] = bp_correct_class(i, row, y[i])

        dJ__dscores /= float(N)

        dJ__dw2 = torch.transpose(h_relu, 0, 1).mm(dJ__dscores)
        dJ__db2 = torch.sum(dJ__dscores, dim=0)
        dJ__dh_relu = dJ__dscores.mm(torch.transpose(w2, 0, 1))
        dJ__dh_relu[h_relu <= 0] = 0 # dJ__dh
        dJ__dh = dJ__dh_relu
        dJ__dw1 = torch.transpose(X, 0, 1).mm(dJ__dh)
        dJ__db1 = torch.sum(dJ__dh, dim=0)

        # Derivative of regularization component
        dJ__dw2 += regularization * w2
        dJ__dw1 += regularization * w1

        # Gradient descent
        w1 -= learning_rate * dJ__dw1
        b1 -= learning_rate * dJ__db1
        w2 -= learning_rate * dJ__dw2
        b2 += -learning_rate * dJ__db2

        # Verbose
        if (epoch == 0) or (epoch % 1000 == 0) or (epoch == num_epochs-1):

            # Accuracy
            score, predicted = torch.max(scores, 1)
            train_accuracy = (y == predicted).sum() / float(N)

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f" %
                (epoch, loss, train_accuracy))

    # Plot trained model
    plot_model(X.numpy(), y.numpy(), w1.numpy(), b1.numpy(), w2.numpy(), b2.numpy())

def pytorch_autograd(X, y, num_hidden_units, num_classes, regularization,
    learning_rate, num_epochs, dtype=torch.FloatTensor):
    """
    Use autograd variables for
    implicit backprop.
    """

    # Dimensions
    N = len(X) # num. samples
    D_in = len(X[0]) # input dim.
    H = num_hidden_units # hidden dim.
    D_out = num_classes # output dim.

    # Convert data to PyTorch tensors
    X = Variable(torch.FloatTensor(X), requires_grad=False)
    y = Variable(torch.LongTensor(y), requires_grad=False)

    # Weights
    w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
    b1 = Variable(torch.zeros((1, H)), requires_grad=True)
    b2 = Variable(torch.zeros((1, D_out)), requires_grad=True)

    for epoch in range(num_epochs):

        # Forward pass
        h = X.mm(w1) + b1.repeat(N,1)
        h_relu = h.clamp(min=0)
        scores = h_relu.mm(w2) + b2.repeat(N,1)
        exp_scores = torch.exp(scores)

        # Softmax normalization
        probs = exp_scores / torch.sum(exp_scores, dim=1).repeat(1,num_classes) # [N, D_out]

        # Cross entropy loss
        y_true_logprobs = -torch.log(torch.gather(probs, 1, y.unsqueeze(1)))
        loss = torch.sum(y_true_logprobs) / float(N)

        # Use autograd to do backprop. This will compute the
        # gradients w.r.t loss for all Variables that have
        # requires_grad=True. So, our w1 and w2 will now have
        # gradient components we can access.
        loss.backward()

        # Update the weights
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data
        b1.data -= learning_rate * b1.grad.data
        b2.data -= learning_rate * b2.grad.data

        # Zero-out the gradients before backprop
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b1.grad.data.zero_()
        b2.grad.data.zero_()

        # Verbose
        if (epoch == 0) or (epoch % 1000 == 0) or (epoch == num_epochs-1):

            # Accuracy
            score, predicted = torch.max(scores, 1)
            train_accuracy = (y.data == predicted.data).sum() / float(N)

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f" %
                (epoch, loss.data[0], train_accuracy))

    # Plot trained model
    plot_model(X.data.numpy(), y.data.numpy(), w1.data.numpy(), b1.data.numpy(),
        w2.data.numpy(), b2.data.numpy())

def pytorch_custom_autograd(X, y, num_hidden_units, num_classes, regularization,
    learning_rate, num_epochs, dtype=torch.FloatTensor):
    """
    Use autograd variables for
    implicit backprop and
    make your own autograd func.
    """

    class ReLU(torch.autograd.Function):
        """
        Just implemnet forward and
        backward pass.
        """

        def forward(self, input_):
            """
            Process inputs and use
            save_for_backward to store
            inputs for backprop.
            """
            self.save_for_backward(input_)
            return input_.clamp(min=0)

        def backward(self, grad_output):
            """
            grad_output is the grad w.r.t loss.
            """
            input_, =self.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input_ < 0] = 0
            return grad_input

    # Dimensions
    N = len(X) # num. samples
    D_in = len(X[0]) # input dim.
    H = num_hidden_units # hidden dim.
    D_out = num_classes # output dim.

    # Convert data to PyTorch tensors
    X = Variable(torch.FloatTensor(X), requires_grad=False)
    y = Variable(torch.LongTensor(y), requires_grad=False)

    # Weights
    w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
    b1 = Variable(torch.zeros((1, H)), requires_grad=True)
    b2 = Variable(torch.zeros((1, D_out)), requires_grad=True)

    for epoch in range(num_epochs):

        # custom autograd func.
        relu = ReLU()

        # Forward pass
        h = X.mm(w1) + b1.repeat(N,1)
        h_relu = relu(h)
        scores = h_relu.mm(w2) + b2.repeat(N,1)
        exp_scores = torch.exp(scores)

        # Softmax normalization
        probs = exp_scores / torch.sum(exp_scores, dim=1).repeat(1,num_classes) # [N, D_out]

        # Cross entropy loss
        y_true_logprobs = -torch.log(torch.gather(probs, 1, y.unsqueeze(1)))
        loss = torch.sum(y_true_logprobs) / float(N)

        # Use autograd to do backprop. This will compute the
        # gradients w.r.t loss for all Variables that have
        # requires_grad=True. So, our w1 and w2 will now have
        # gradient components we can access.
        loss.backward()

        # Update the weights
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data
        b1.data -= learning_rate * b1.grad.data
        b2.data -= learning_rate * b2.grad.data

        # Zero-out the gradients before backprop
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b1.grad.data.zero_()
        b2.grad.data.zero_()

        # Verbose
        if (epoch == 0) or (epoch % 1000 == 0) or (epoch == num_epochs-1):

            # Accuracy
            score, predicted = torch.max(scores, 1)
            train_accuracy = (y.data == predicted.data).sum() / float(N)

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f" %
                (epoch, loss.data[0], train_accuracy))

    # Plot trained model
    plot_model(X.data.numpy(), y.data.numpy(), w1.data.numpy(), b1.data.numpy(),
        w2.data.numpy(), b2.data.numpy())

def pytorch_nn(X, y, num_hidden_units, num_classes, regularization,
    learning_rate, num_epochs, dtype=torch.FloatTensor):
    """
    Use nn.Module.
    """

    # Dimensions
    N = len(X) # num. samples
    D_in = len(X[0]) # input dim.
    H = num_hidden_units # hidden dim.
    D_out = num_classes # output dim.

    # Convert data to PyTorch tensors
    X = Variable(torch.FloatTensor(X), requires_grad=False)
    y = Variable(torch.LongTensor(y), requires_grad=False)

    # Model
    class Model(nn.Module):
        """
        NN model using nn.Module
        """
        def __init__(self):
            """
            Initialize weights.
            """
            super(Model, self).__init__()
            self.fc1 = nn.Linear(D_in, H)
            self.fc2 = nn.Linear(H, D_out)

        def forward(self, x):
            """
            Forward pass.
            """
            z = F.relu(self.fc1(x))
            z = self.fc2(z)
            return z

    # Create model
    model = Model()

    # Objective
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(num_epochs):

        # Zero-out gradients
        model.zero_grad()

        # Forward pass
        scores = model(X) # logits

        # Loss
        loss = criterion(scores, y)

        # Use autograd to do backprop. This will compute the
        # gradients w.r.t loss for all Variables that have
        # requires_grad=True. So, our w1 and w2 will now have
        # gradient components we can access.
        loss.backward()

        # Update the weights
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data

        # Verbose
        if (epoch == 0) or (epoch % 1000 == 0) or (epoch == num_epochs-1):

            # Accuracy
            score, predicted = torch.max(scores, 1)
            train_accuracy = (y.data == predicted.data).sum() / float(N)

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f" %
                (epoch, loss.data[0], train_accuracy))

    # Plot trained model
    plot_model(X.data.numpy(), y.data.numpy(),
        torch.transpose(model.fc1.weight.data, 0, 1).numpy(),
        model.fc1.bias.data.numpy(),
        torch.transpose(model.fc2.weight.data, 0, 1).numpy(),
        model.fc2.bias.data.numpy())

def pytorch_optimization(X, y, num_hidden_units, num_classes, regularization,
    learning_rate, num_epochs, dtype=torch.FloatTensor):
    """
    Putting it all together
    with the optimization tools.
    """

    # Dimensions
    N = len(X) # num. samples
    D_in = len(X[0]) # input dim.
    H = num_hidden_units # hidden dim.
    D_out = num_classes # output dim.

    # Convert data to PyTorch tensors
    X = Variable(torch.FloatTensor(X), requires_grad=False)
    y = Variable(torch.LongTensor(y), requires_grad=False)

    # Model
    class Model(nn.Module):
        """
        NN model using nn.Module
        """
        def __init__(self):
            """
            Initialize weights.
            """
            super(Model, self).__init__()
            self.fc1 = nn.Linear(D_in, H)
            self.fc2 = nn.Linear(H, D_out)

        def forward(self, x):
            """
            Forward pass.
            """
            z = F.relu(self.fc1(x))
            z = self.fc2(z)
            return z

    # Create model
    model = Model()

    # Objective
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):

        # Forward pass
        scores = model(X) # logits

        # Loss
        loss = criterion(scores, y)

        # Use autograd to do backprop. This will compute the
        # gradients w.r.t loss for all Variables that have
        # requires_grad=True. So, our w1 and w2 will now have
        # gradient components we can access.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verbose
        if (epoch == 0) or (epoch % 1000 == 0) or (epoch == num_epochs-1):

            # Accuracy
            score, predicted = torch.max(scores, 1)
            train_accuracy = (y.data == predicted.data).sum() / float(N)

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f" %
                (epoch, loss.data[0], train_accuracy))

    # Plot trained model
    plot_model(X.data.numpy(), y.data.numpy(),
        torch.transpose(model.fc1.weight.data, 0, 1).numpy(),
        model.fc1.bias.data.numpy(),
        torch.transpose(model.fc2.weight.data, 0, 1).numpy(),
        model.fc2.bias.data.numpy())

def pytorch_cuda(X, y, num_hidden_units, num_classes, regularization,
    learning_rate, num_epochs, cuda_enabled):
    """
    CUDA enabled.
    """

    # dtype
    if not cuda_enabled:
        dtype = torch.FloatTensor
        print ("Not using GPU :(")
    else:
        dtype = torch.cuda.FloatTensor
        print ("Using GPU :)")

    # Convert data to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    if cuda_enabled:
        X = X.cuda()
        y = y.cuda()
    X = Variable(X, requires_grad=False)
    y = Variable(y, requires_grad=False)

    # Dimensions
    N = len(X) # num. samples
    D_in = len(X[0]) # input dim.
    H = num_hidden_units # hidden dim.
    D_out = num_classes # output dim.

    # Model
    class Model(nn.Module):
        """
        NN model using nn.Module
        """
        def __init__(self):
            """
            Initialize weights.
            """
            super(Model, self).__init__()
            self.fc1 = nn.Linear(D_in, H)
            self.fc2 = nn.Linear(H, D_out)

        def forward(self, x):
            """
            Forward pass.
            """
            z = F.relu(self.fc1(x))
            z = self.fc2(z)
            return z

    # Create model
    model = Model()
    if cuda_enabled:
        model.cuda()

    # Objective
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):

        # Forward pass
        scores = model(X) # logits

        # Loss
        loss = criterion(scores, y)

        # Use autograd to do backprop. This will compute the
        # gradients w.r.t loss for all Variables that have
        # requires_grad=True. So, our w1 and w2 will now have
        # gradient components we can access.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verbose
        if (epoch == 0) or (epoch % 1000 == 0) or (epoch == num_epochs-1):

            # Accuracy
            score, predicted = torch.max(scores, 1)
            train_accuracy = (y.data == predicted.data).cpu().sum() / float(N)

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f" %
                (epoch, loss.data[0], train_accuracy))

    # Plot trained model
    plot_model(X.data.numpy(), y.data.numpy(),
        torch.transpose(model.fc1.weight.data, 0, 1).numpy(),
        model.fc1.bias.data.numpy(),
        torch.transpose(model.fc2.weight.data, 0, 1).numpy(),
        model.fc2.bias.data.numpy())

def main(FLAGS):
    """
    """

    # Create the data
    X, y = get_data (FLAGS.seed, FLAGS.num_samples, FLAGS.dimensions,
        FLAGS.num_classes)
    #plot_data(X, y)

    # Numpy
    #numpy_version(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
    #    FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs)

    # PyTorch tensors (no autograd, opti, etc.)
    #pytorch_tensors(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
    #    FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs)

    # PyTorch autograd
    #pytorch_autograd(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
    #    FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs)

    # PyTorch custom autograd
    #pytorch_custom_autograd(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
    #    FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs)

    # Pytorch nn.Module
    #pytorch_nn(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
    #    FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs)

    # PyTorch optim
    #pytorch_optimization(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
    #    FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs)

    # PyTorch cuda
    pytorch_cuda(X, y, FLAGS.num_hidden_units, FLAGS.num_classes,
        FLAGS.regularization, FLAGS.learning_rate, FLAGS.num_epochs,
        cuda_enabled=FLAGS.cuda)

if __name__ == '__main__':

    FLAGS = get_args()

    # CUDA
    FLAGS.cuda = not FLAGS.no_cuda and torch.cuda.is_available()

    # Seeding
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed_all(FLAGS.seed)

    main(FLAGS)





