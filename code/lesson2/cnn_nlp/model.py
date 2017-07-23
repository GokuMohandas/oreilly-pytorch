"""
model.py
"""
from __future__ import (
    print_function,
)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import (
    Variable,
)
from torch.nn import (
    init,
)

class CNN(nn.Module):
    """
    """
    def __init__(self, num_channels, num_filters, num_classes,
        dropout_p):
        """
        Initialize weights.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(2, num_channels),
            stride=1,
            padding=0,
            bias=True,
            )
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(3, num_channels),
            stride=1,
            padding=0,
            bias=True,
            )
        self.conv3 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(4, num_channels),
            stride=1,
            padding=0,
            bias=True,
            )
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(3*num_filters, num_classes)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights.
        """
        for conv in [self.conv1, self.conv2, self.conv3]:
            init.xavier_uniform(conv.weight, gain=1)
            init.constant(conv.bias, 0.1)


    def forward(self, x):
        """
        Forward pass.
        """

        # Make each input 3D to apply filters
        x = x.unsqueeze(1) # (N, 1, W, D)

        # Apply convolution and pooling
        convs = [self.conv1, self.conv2, self.conv3]
        x = [F.relu(conv(x)).squeeze(3) for conv in convs] # num-of-unique-filter-sizes X [N, num_kernels, #words]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # num-of-unique-filter-sizes X [N, num_kernels]

        # Concat activations
        x = torch.cat(x, 1) # [N, len(filter_lengths)*num_kernels]

        # Dropout
        x = self.dropout(x)

        # FC layer
        logits = self.fc1(x)

        return logits


