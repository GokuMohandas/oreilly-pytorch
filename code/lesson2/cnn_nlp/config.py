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
data_arg.add_argument('--data_dir',        type=str,                             help='location of data files (default: data/ag_news_csv).')
data_arg.add_argument('--split_ratio',     type=float,          default=0.8,     help='train/test split ratio.')

# Utility parameters
util_arg = add_argument_group('Utilities')
util_arg.add_argument('--mode',            type=str,            default='infer', help='train|infer')
util_arg.add_argument('--model_name',      type=str,                             help='model name.')
util_arg.add_argument('--num_epochs',      type=int,            default=30,     help='number of epochs to train.')
util_arg.add_argument('--batch_size',      type=int,            default=64,      help='batch size.')
util_arg.add_argument('--no_cuda',         action='store_true', default=False,   help='enable CUDA training on K80s')

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--lr',             type=float,          default=1e-3,    help='learning rate')
model_arg.add_argument('--num_filters',    type=int,            default=200,     help='number of each type of filter length.')
model_arg.add_argument('--dropout_p',      type=float,          default=0.5,     help='dropout.')
model_arg.add_argument('--decay_rate',     type=float,          default=0.9,     help='decay_rate.')
model_arg.add_argument('--max_grad_norm',  type=float,          default=5.0,     help='clip the gradient to prevent explosion.')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
