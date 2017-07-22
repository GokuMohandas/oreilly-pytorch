"""
train.py
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

from tqdm import (
    tqdm,
)

from pycrayon import (
    CrayonClient,
)

from config import (
    basedir,
)

from utils import (
    generate_epoch,
    pad,
)

from model import (
    CNN,
)

# Connect to the server for Crayon (tensorboard)
cc = CrayonClient(hostname="localhost", port=8889)

def process_batch(exp, model, inputs, targets, num_samples,
    criterion, optimizer=None, max_grad_norm=None, is_training=True):
    """
    Process a batch of samples with the model.
    """

    # Mode
    if is_training:
        model.train()
    else:
        model.eval()

    # Train batch
    scores = model(inputs)

    # Loss
    loss = criterion(scores, targets)

    if is_training:
        # Use autograd to do backprop. This will compute the
        # gradients w.r.t loss for all Variables that have
        # requires_grad=True. So, our w1 and w2 will now have
        # gradient components we can access.
        optimizer.zero_grad()
        loss.backward()

        # Clip the gradient norms
        nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)

        # Update params
        optimizer.step()

    # Accuracy
    score, predicted = torch.max(scores, 1)
    accuracy = (targets.data == predicted.data).sum() / float(num_samples)

    return loss, accuracy

def train(data_dir, char2index, train_data, test_data, num_epochs, batch_size,
    num_filters, learning_rate, decay_rate, max_grad_norm, dropout_p):
    """
    """

    # Create a new experiment
    exp_name = "%s_%.2E_%.6f_%.2f" % (
        data_dir.split("/")[-1], learning_rate, decay_rate, dropout_p)
    try:
        cc.remove_experiment(exp_name)
        exp = cc.create_experiment(exp_name)
    except:
        exp = cc.create_experiment(exp_name)

    # index2class
    with open(os.path.join(basedir, data_dir, 'index2class.json'), 'r') as f:
        index2class = json.load(f)

    # Model
    model = CNN(
        num_channels=len(char2index),
        num_filters=num_filters,
        num_classes=len(index2class),
        dropout_p=dropout_p,
        )

    # Objective
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for num_train_epoch, epoch in enumerate(
        generate_epoch(train_data, num_epochs, batch_size)):

        # Timer
        start = time.time()

        # Decay learning rate
        learning_rate = learning_rate * (decay_rate ** (num_train_epoch // 1.0))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        train_batch_loss = 0.0
        train_batch_accuracy = 0.0

        for train_batch_num, (inputs, targets, true_batch_size) in enumerate(epoch):

            # Pad the ordered inputs
            inputs, _ = pad(
                inputs=inputs,
                char2index=char2index,
                max_length=len(inputs[0]),
                )

            # Convert to Variable
            inputs = Variable(torch.FloatTensor(inputs), requires_grad=False)
            targets = Variable(torch.LongTensor(targets), requires_grad=False)

            loss, accuracy = process_batch(
                exp=exp,
                model=model,
                inputs=inputs,
                targets=targets,
                num_samples=true_batch_size,
                criterion=criterion,
                optimizer=optimizer,
                max_grad_norm=max_grad_norm,
                is_training=True,
                )

            # Add to batch scalars
            train_batch_loss += loss.data[0]
            train_batch_accuracy += accuracy

            # Record metrics
            exp.add_scalar_value("train_loss", value=loss.data[0])
            exp.add_scalar_value("train_accuracy", value=accuracy)

        # Testing
        for num_test_epoch, epoch in enumerate(
            generate_epoch(test_data, 1, batch_size)):

            test_batch_loss = 0.0
            test_batch_accuracy = 0.0

            for test_batch_num, (inputs, targets, true_batch_size) in enumerate(epoch):

                # Pad the ordered inputs
                inputs, _ = pad(
                    inputs=inputs,
                    char2index=char2index,
                    max_length=len(inputs[0]),
                    )

                # Convert to Variable
                inputs = Variable(torch.FloatTensor(inputs), requires_grad=False)
                targets = Variable(torch.LongTensor(targets), requires_grad=False)

                loss, accuracy = process_batch(
                    exp=exp,
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    num_samples=true_batch_size,
                    criterion=criterion,
                    is_training=False,
                    )

                # Add to batch scalars
                test_batch_loss += loss.data[0]
                test_batch_accuracy += accuracy

                # Record metrics
                exp.add_scalar_value("test_loss", value=loss.data[0])
                exp.add_scalar_value("test_accuracy", value=accuracy)


        if (num_train_epoch==0) or (num_train_epoch%1 == 0) or \
           (num_train_epoch == num_epochs-1):

           # Verbose
            time_remain = (time.time() - start) * (num_epochs - (num_train_epoch+1))
            minutes = time_remain // 60
            seconds = time_remain - minutes*60
            print ("TIME REMAINING: %im %is" % (minutes, seconds))
            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f, [TEST LOSS]: %.6f, [TEST ACC]: %.3f" % (
                num_train_epoch,
                train_batch_loss/float(train_batch_num+1),
                train_batch_accuracy/float(train_batch_num+1),
                test_batch_loss/float(test_batch_num+1),
                test_batch_accuracy/float(test_batch_num+1)))

    # Save the model
    torch.save(model, os.path.join(
        basedir, "data", data_dir.split("/")[-1], "model-%s.pt"%(exp_name)))




