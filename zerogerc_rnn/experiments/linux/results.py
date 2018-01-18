import os

import numpy as np
import torch
from experiments.linux.constants import alphabet
from experiments.linux.lstm import LSTMLinuxNetwork
from lib.utils.state import load_if_saved
from torch.autograd import Variable

from global_constants import ROOT_DIR
from zerogerc_rnn.lib.data.character import Corpus

SEQ_LEN = 100
BATCH_SIZE = 100

HIDDEN_SIZE = 64
NUM_LAYERS = 1

DATA_PATH = ROOT_DIR + '/data/linux_kernel_mini.txt'


def pick_single_input(data_tensor, start, ntokens):  # one hot encoding of data
    positions = data_tensor[np.arange(start, start + SEQ_LEN), :]
    one_hot = torch.zeros(SEQ_LEN, ntokens)
    one_hot.scatter_(1, positions.view(SEQ_LEN, 1), 1.)
    return one_hot


def pick_single_target(data_tensor, start):  # index of target (input shifted by one)
    return data_tensor[np.arange(start + 1, start + 1 + SEQ_LEN), :]


def pick_batch(data_tensor, start_indexes, single_picker):
    # call chunk_pick and add one dimension for batching
    out = [single_picker(data_tensor, start * SEQ_LEN).unsqueeze(1) for start in start_indexes]
    return torch.cat(out, dim=1)


def read_data():
    start = 1023
    corpus = Corpus(os.path.join(ROOT_DIR, 'data'), single_file='kernel_concatenated/test.txt', letters=alphabet)
    ntokens = len(corpus.all_letters)

    input_tensor = Variable(pick_single_input(corpus.single.unsqueeze(1), start, ntokens).unsqueeze(1))
    target_tensor = pick_single_target(corpus.single.unsqueeze(1), start)

    INPUT_SIZE = len(corpus.all_letters)
    OUTPUT_SIZE = len(corpus.all_letters)

    network = LSTMLinuxNetwork(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS
    )

    load_if_saved(network, 'linux')

    output_tensor = network(input_tensor)

    print_string(corpus, torch.max(output_tensor.squeeze(1), 1)[1].data, tag='Predicted')
    print_string(corpus, target_tensor.view(-1), tag='Actual')

def print_string(corpus, letter_positions, tag=None):
    s = ""
    for lp in letter_positions:
        s += corpus.idx2char[lp]

    if tag is None:
        print(s)
    else:
        print('{}: {}'.format(tag, s))


if __name__ == '__main__':
    read_data()