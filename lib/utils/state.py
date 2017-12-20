import os
import torch

from constants import ROOT_DIR

BASE_PATH = ROOT_DIR + '/saved/'


def load_if_saved(model, path='temp'):
    """
    Loads state of the model if previously saved.
    """
    if os.path.isfile(BASE_PATH + path):
        model.load_state_dict(torch.load(BASE_PATH + path))


def save_model(model, path='temp'):
    """
    Saves state of the model by specified path.
    """
    torch.save(model.state_dict(), BASE_PATH + path)
