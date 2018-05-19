from __future__ import unicode_literals, print_function, division

import numpy as np
import os
from io import open

import torch

DEFAULT_ENCODING = 'ISO-8859-1'


def read_lines(filename, encoding=DEFAULT_ENCODING):
    """
    Read a file and split into lines
    """
    """Read first *limit* lines from file and returns list of them."""
    lines = []
    with open(filename, mode='r', encoding=encoding) as f:
        for l in f:
            if l[-1] == '\n':
                lines.append(l[:-1])
            else:
                lines.append(l)
    return lines


def load_if_saved(model, path):
    """Loads state of the model if previously saved."""
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        print('Model restored from file.')
    else:
        raise Exception('Model file not exists File: {}'.format(path))


def load_cuda_on_cpu(model, path):
    """Loads CUDA model for testing on non CUDA device."""
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        print('Model restored from file.')
    else:
        raise Exception('Model file not exists. File: {}'.format(path))


def save_model(model, path):
    """Saves state of the model by specified path."""
    torch.save(model.state_dict(), path)
