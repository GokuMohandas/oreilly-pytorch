"""
Main
"""
from __future__ import (
    print_function,
)

import os

from config import (
    basedir,
    get_config,
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

    if FLAGS.mode == "train":
        train(FLAGS)
    elif FLAGS.mode == "infer":
        infer(FLAGS)
    else:
        raise Exception("Choose --mode=<train|infer>")

if __name__ == '__main__':
    FLAGS, _ = get_config()
    main(FLAGS)