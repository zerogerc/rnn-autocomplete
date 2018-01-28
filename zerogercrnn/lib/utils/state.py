import os

import torch

def load_if_saved(model, path):
    """Loads state of the model if previously saved."""
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))


def save_model(model, path):
    """Saves state of the model by specified path."""
    torch.save(model.state_dict(), path)
