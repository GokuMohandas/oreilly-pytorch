"""
train.py
Training the embeddings.
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
import sys
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

from data import (
    process_data,
)

from utils import (
    generate_epoch,
)

from model import (
    MLP,
)

def process_batch(
    model, inputs, targets, num_samples, criterion,
    optimizer=None, max_grad_norm=None, is_training=True):
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

def training_procedure(
    model, criterion, optimizer,
    train_loader, test_loader, num_epochs,
    learning_rate, decay_rate, max_grad_norm):
    """
    """

    # Train
    for num_train_epoch, epoch in enumerate(
        generate_epoch(train_loader, num_epochs)):

        # Timer
        start = time.time()

        # Decay learning rate
        learning_rate = learning_rate * (decay_rate ** (num_train_epoch // 1.0))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        train_batch_loss = 0.0
        train_batch_accuracy = 0.0

        for train_batch_num, data in enumerate(epoch):

            # Convert to Variable
            inputs = Variable(data[0], requires_grad=False)
            targets = Variable(data[1], requires_grad=False)

            loss, accuracy = process_batch(
                model=model,
                inputs=inputs,
                targets=targets,
                num_samples=len(inputs),
                criterion=criterion,
                optimizer=optimizer,
                max_grad_norm=max_grad_norm,
                is_training=True,
                )

            # Add to batch scalars
            train_batch_loss += loss.data[0]
            train_batch_accuracy += accuracy
        # Testing
        for num_test_epoch, epoch in enumerate(
            generate_epoch(test_loader, num_epochs=1)):

            test_batch_loss = 0.0
            test_batch_accuracy = 0.0

            for test_batch_num, data in enumerate(epoch):

                # Convert to Variable
                inputs = Variable(data[0], volatile=True, requires_grad=False)
                targets = Variable(data[1], requires_grad=False)

                loss, accuracy = process_batch(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    num_samples=len(inputs),
                    criterion=criterion,
                    is_training=False,
                    )

                # Add to batch scalars
                test_batch_loss += loss.data[0]
                test_batch_accuracy += accuracy

        if (num_train_epoch==0) or (num_train_epoch%1 == 0) or \
           (num_train_epoch == num_epochs-1):

           # Verbose
            time_remain = (time.time() - start) * (num_epochs - (num_train_epoch+1))
            minutes = time_remain // 60
            seconds = time_remain - minutes*60
            print ("[TIME REMAINING]: %im %is " % (minutes, seconds))
            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f, [TEST LOSS]: %.6f, [TEST ACC]: %.3f " % (
                num_train_epoch,
                train_batch_loss/float(train_batch_num+1),
                train_batch_accuracy/float(train_batch_num+1),
                test_batch_loss/float(test_batch_num+1),
                test_batch_accuracy/float(test_batch_num+1)))
            sys.stdout.flush()

    return model

def train(FLAGS):
    """
    Train our embeddings.
    """

    # Get data loaders
    print ("==> Reading and processing the data ... ", end="")
    train_loader, test_loader, num_unique_words = process_data(
        data_dir=FLAGS.data_dir,
        data_file=FLAGS.data_file,
        vocab_size=FLAGS.vocab_size,
        window_size=FLAGS.window_size,
        split_ratio=FLAGS.split_ratio,
        batch_size=FLAGS.batch_size,
        )
    print ("[COMPLETE]")

    # Initialize model, criterion, loss
    print ("==> Initializing model components ... ", end="")
    model = MLP(
        D_in=num_unique_words,
        H=FLAGS.embedding_dim,
        )
    # Objective
    criterion = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    print ("[COMPLETE]")

    # Train the model
    print ("==> Training the model ... [IN PROGRESS]")
    model = training_procedure(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=FLAGS.num_epochs,
        learning_rate=FLAGS.lr,
        decay_rate=FLAGS.decay_rate,
        max_grad_norm=FLAGS.max_grad_norm,
        )
    print ("\n[COMPLETE]")

    # Save the model
    print ("==> Saving the model ... [IN PROGRESS]")
    torch.save(model, os.path.join(basedir, FLAGS.data_dir, "model.pt"))
    print ("\n[COMPLETE]")










