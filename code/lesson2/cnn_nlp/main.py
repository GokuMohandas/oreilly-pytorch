"""
Main
"""
from __future__ import (
    print_function,
)

import os
import numpy as np
import json

from config import (
    basedir,
    get_config,
)

from data import (
    main as process_data,
)

from utils import (
    sample,
)

from train import (
    train,
)

from infer import (
    infer,
)

def main(FLAGS):
    """
    """

    if FLAGS.mode == 'train':

        # Process the data
        train_data, test_data = process_data(
            data_dir=FLAGS.data_dir,
            split_ratio=FLAGS.split_ratio,
            )

        # Sample
        sample(
            data=train_data,
            data_dir=FLAGS.data_dir,
            )

        # Load components
        with open(os.path.join(basedir, FLAGS.data_dir, 'char2index.json'), 'r') as f:
            char2index = json.load(f)

        # Training
        train(
            data_dir=FLAGS.data_dir,
            char2index=char2index,
            train_data=train_data,
            test_data=test_data,
            num_epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size,
            num_filters=FLAGS.num_filters,
            learning_rate=FLAGS.lr,
            decay_rate=FLAGS.decay_rate,
            max_grad_norm=FLAGS.max_grad_norm,
            dropout_p=FLAGS.dropout_p,
            )

    elif FLAGS.mode == 'infer':

        # Inference
        infer(
            data_dir=FLAGS.data_dir,
            model_name=FLAGS.model_name,
            )

    else:
        raise Exception('Choose --mode train|infer')


if __name__ == '__main__':
    FLAGS, _ = get_config()
    main(FLAGS)