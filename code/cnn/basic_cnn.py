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
    models,
    transforms,
)

from torch.autograd import (
    Variable,
)

from torchvision import (
    datasets,
    transforms,
)

basedir = os.path.abspath(os.path.dirname(__file__))

def get_args():
    """
    Arguments
    """
    parser = argparse.ArgumentParser (description='Main arguments')
    parser.add_argument ('--model_type', type=str, default='finetuned',
        help='naive|finetuned|feature_extractor')
    parser.add_argument ('--seed', type=int, default=1234)
    parser.add_argument ('--num_workers', type=int, default=1) # load data
    parser.add_argument ('--num_epochs', type=int, default=2)
    parser.add_argument ('--batch_size', type=int, default=4)
    parser.add_argument ('--learning_rate', type=float, default=1e-4)
    parser.add_argument ('--log_interval', type=int, default=1)
    parser.add_argument ('--no_cuda', action='store_true', default=False)
    parser.add_argument ('--pretrained', action='store_true', default=False)

    return parser.parse_args ()

def load_data(batch_size, num_workers):
    """
    Load data from torchvision.
    Format:

    data/
        train/
            class1/
                image1
                image2
                ...
            class2/
                image1
                image2
                ...
            ...
        test/
            class1/
                image1
                image2
                ...
            class2/
                image1
                image2
                ...
            ...
    """
    data_dir = 'data'
    train_dir = os.path.join(basedir, data_dir, 'train')
    val_dir = os.path.join(basedir, data_dir, 'test')

    # Data normalizeion and augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]),
        'test': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            ])
    }

    # Create loaders from the images in our folders
    data_sets = {x: datasets.ImageFolder(train_dir, data_transforms[x])
        for x in ['train', 'test']}
    dataset_loaders = {x: torch.utils.data.DataLoader(
        data_sets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        ) for x in ['train', 'test']}
    dataset_sizes = {x: len(data_sets[x]) for x in ['train', 'test']}
    dataset_classes = data_sets['train'].classes

    print (dataset_sizes)
    print (dataset_classes)

    # Sample a few images
    images, classes = next(iter(dataset_loaders['train']))
    grid_images = torchvision.utils.make_grid(images)
    plt.imshow(grid_images.numpy().transpose((1,2,0)))
    plt.show()

    return dataset_loaders, dataset_classes

class CNN(nn.Module):
    """
    Simple CNN for cat-dog data.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=11,
            stride=4,
            )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=11,
            stride=4,
            )
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(2*2*128, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        """
        Forward pass.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(
            input=self.conv2_drop(self.conv2(x)),
            kernel_size=2,
            stride=2,
            ))
        x = x.view(-1, 2*2*128) # flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

def train(model_type, dataset_loaders, num_epochs, learning_rate,
    log_interval, cuda_enabled):
    """
    """

    if model_type == 'naive':
        # Model
        model = CNN()
    elif model_type == 'finetuned':
        # Finetuned resnet18
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
    elif model_type == 'feature_extractor':
        # Pretrained resnet as feature extractor
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False # freeze weights
        # Add new fc layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)


    if model_type == 'feature_extractor':
        # Optimizer (only update fc)
        optimizer = torch.optim.Adam(
            model.fc.parameters(), lr=learning_rate)
    else:
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate)

    if cuda_enabled:
        model.cuda()

    # Objective
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_train_batches = 0

        for batch, (X, y) in enumerate(dataset_loaders['train']):

            num_train_batches += 1
            if cuda_enabled:
                X, y = X.cuda(), y.cuda()
            X, y = Variable(X, requires_grad=False), Variable(y, requires_grad=False)

            # Forward pass
            scores = model(X) # logits

            # Loss
            loss = criterion(scores, y)
            train_loss += loss.data[0] / float(len(X))

            # Accuracy
            score, predicted = torch.max(scores, 1)
            train_acc += (y.data == predicted.data).sum() / float(len(X))

            # Use autograd to do backprop. This will compute the
            # gradients w.r.t loss for all Variables that have
            # requires_grad=True.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verbose
        if (epoch == 0) or (epoch % log_interval == 0) or (epoch == num_epochs-1):

            # Testing
            model.eval()
            test_loss = 0.0
            test_acc = 0.0
            num_test_batches = 0
            for batch, (X, y) in enumerate(dataset_loaders['test']):
                num_test_batches += 1
                if cuda_enabled:
                    X, y = X.cuda(), y.cuda()
                X, y = Variable(X, requires_grad=False), Variable(y, requires_grad=False)

                # Forward pass
                scores = model(X) # logits

                # Loss
                loss = criterion(scores, y)
                test_loss += loss.data[0] / float(len(X))

                # Accuracy
                score, predicted = torch.max(scores, 1)
                test_acc += (y.data == predicted.data).sum() / float(len(X))

            print ("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACC]: %.3f, [TEST LOSS]: %.6f, [TEST ACC]: %.3f" % (
                epoch,
                train_loss/float(num_train_batches),
                train_acc/float(num_train_batches),
                test_loss/float(num_test_batches),
                test_acc/float(num_test_batches),
                ))

    # Save the model
    torch.save(model, os.path.join(
        basedir, 'data', 'model.pt'))

def infer(dataset_loaders, dataset_classes):
    """
    Inference of a sample image.
    """

    # Load model
    model = torch.load(os.path.join(
         basedir, 'data', 'model.pt'))

    # Infer
    images, labels = next(iter(dataset_loaders['train']))
    sample_image, sample_label = images[0], labels[0]
    plt.imshow(sample_image.numpy().transpose((1,2,0)))
    score, y_pred = torch.max(model(Variable(sample_image.unsqueeze(0))), 1)
    plt.title("[TARGET]: %s, [PREDICTION]: %s" % (
        dataset_classes[sample_label], dataset_classes[y_pred.data[0][0]]))
    plt.show()


def main(FLAGS):
    """
    """

    # Load data loaders
    dataset_loaders, dataset_classes = load_data(
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        )
    '''
    train(
        model_type=FLAGS.model_type,
        dataset_loaders=dataset_loaders,
        num_epochs=FLAGS.num_epochs,
        learning_rate=FLAGS.learning_rate,
        log_interval=FLAGS.log_interval,
        cuda_enabled=FLAGS.cuda,
        )
    '''

    infer(
        dataset_loaders=dataset_loaders,
        dataset_classes=dataset_classes,
        )

if __name__ == '__main__':

    FLAGS = get_args()

    # CUDA
    FLAGS.cuda = not FLAGS.no_cuda and torch.cuda.is_available()

    # Seeding
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed_all(FLAGS.seed)

    main(FLAGS)









