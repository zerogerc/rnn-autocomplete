import os

import torch


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
