import os

import numpy as np
import torch
from torch.autograd import Variable

from zerogercrnn.experiments.linux.data import alphabet
from zerogercrnn.experiments.linux.data import read_data_mini
from zerogercrnn.experiments.linux.lstm import LSTMLinuxNetwork
from zerogercrnn.lib.data.character import create_char_to_idx_and_backward
from zerogercrnn.lib.utils.state import load_if_saved
from zerogercrnn.lib.visualization.text import show_diff

SEQ_LEN = 100

HIDDEN_SIZE = 256
NUM_LAYERS = 2


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

    batcher, corpus = read_data_mini(
        single=os.path.join(os.getcwd(), 'data_dir/linux_kernel_mini.txt'),
        alphabet=alphabet,
        seq_len=SEQ_LEN
    )

    data_generator = batcher.data_map['test'].get_batched_random(batch_size=1)

    input_tensor, target_tensor = next(data_generator)

    INPUT_SIZE = len(corpus.alphabet)
    OUTPUT_SIZE = len(corpus.alphabet)

    network = LSTMLinuxNetwork(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS
    )

    load_if_saved(network, os.path.join(os.getcwd(), 'saved_models/model_epoch_10'))

    output = network(Variable(input_tensor))

    predicted = convert(corpus, torch.max(output.squeeze(1), 1)[1].data)
    actual = convert(corpus, target_tensor.view(-1))

    show_diff(text=predicted, actual=actual)


def convert(corpus, letter_positions):
    char2idx, idx2char = create_char_to_idx_and_backward(alphabet=corpus.alphabet)
    s = ''
    for lp in letter_positions:
        s += idx2char[lp]

    return s


if __name__ == '__main__':
    read_data()
