"""
config.py
"""

import os
import argparse

basedir = os.path.abspath(os.path.dirname(__file__))
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg

# Data parameters
data_arg = add_argument_group('Data')
data_arg.add_argument(
    '--data_dir',
    type=str,
    default='data/oz',
    help='location of data files (default: data/oz).',
    )
data_arg.add_argument(
    '--data_file',
    type=str,
    default='oz.txt',
    help='location of data files (default: oz.txt).',
    )
data_arg.add_argument(
    '--vocab_size',
    type=str,
    default=5000,
    help='Embeddings for top <vocab_size> words. Rest are UNK.',
    )
data_arg.add_argument(
    '--window_size',
    type=int,
    default=1,
    help='Window size for Skip-gram model (default: 1).',
    )
data_arg.add_argument(
    '--split_ratio',
    type=float,
    default=0.8,
    help='Percentage of data to use for training (default: 0.8).',
    )



# Utility parameters
util_arg = add_argument_group('Utilities')
util_arg.add_argument(
    '--mode',
    type=str,
    required=True,
    help='train|infer',
    )
util_arg.add_argument(
    '--num_epochs',
    type=int,
    default=25,
    help='number of epochs to train (default: 25).',
    )
util_arg.add_argument(
    '--batch_size',
    type=int,
    default=512,
    help='batch size (default: 5).',
        )
util_arg.add_argument(
    '--no_cuda',
    action='store_true',
    default=False,
    help='enable CUDA training on K80s',
    )

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='learning rate',
    )
model_arg.add_argument(
    '--embedding_dim',
    type=int,
    default=100,
    help='embedding dimension (default: 100).',
    )
model_arg.add_argument(
    '--decay_rate',
    type=float,
    default=0.9999,
    help='decay_rate.',
    )
model_arg.add_argument(
    '--max_grad_norm',
    type=float,
    default=25.0,
    help='clip the gradient to prevent explosion.',
    )

# Data parameters
infer_arg = add_argument_group('Inference')
infer_arg.add_argument(
    '--TOP_K',
    type=int,
    default=10,
    help='Top K similar words for a given word (default: 10).',
    )

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed