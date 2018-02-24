import os

import numpy as np
import torch
from torch.autograd import Variable

from zerogercrnn.experiments.linux.data import alphabet, read_data
from zerogercrnn.experiments.linux.models.lstm import LSTMLinuxNetwork
from zerogercrnn.lib.data.character import create_char_to_idx_and_backward, tokenize
from zerogercrnn.lib.utils.state import load_if_saved
from zerogercrnn.lib.visualization.text import show_diff

SEQ_LEN = 10000

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


def visualize_for_input_target(input_tensor, target_tensor, alphabet):
    INPUT_SIZE = len(alphabet)
    OUTPUT_SIZE = len(alphabet)

    network = LSTMLinuxNetwork(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS
    )

    load_if_saved(network, os.path.join(os.getcwd(), 'saved_models/model_epoch_10'))

    output = network(Variable(input_tensor))

    predicted = convert(torch.max(output.squeeze(1), 1)[1].data, alphabet=alphabet)
    actual = convert(target_tensor.view(-1), alphabet=alphabet)

    show_diff(text=predicted, actual=actual, file=os.path.join(os.getcwd(), 'result.html'))


def convert(letter_positions, alphabet):
    char2idx, idx2char = create_char_to_idx_and_backward(alphabet=alphabet)
    s = ''
    for lp in letter_positions:
        s += idx2char[lp]
    return s


def visualize_for_test():
    # batcher, corpus = read_data_mini(
    #     single=os.path.join(os.getcwd(), 'data_dir/linux_kernel_mini.txt'),
    #     alphabet=alphabet,
    #     seq_len=SEQ_LEN
    # )

    batcher, corpus = read_data(
        datadir=os.path.join(os.getcwd(), 'data_dir/kernel_concat'),
        seq_len=SEQ_LEN
    )

    data_generator = batcher.data_map['test'].get_batched_random(batch_size=1)

    input_tensor, target_tensor = next(data_generator)
    visualize_for_input_target(input_tensor, target_tensor, alphabet=corpus.alphabet)


def visualize_for_source_file(path):
    data = tokenize(path=path, alphabet=alphabet).unsqueeze(1) # tensor of size [Nx1]

    input_tensor = pick_single_input(data_tensor=data, start=0, ntokens=len(alphabet))
    target_tensor = pick_single_target(data_tensor=data, start=0)

    visualize_for_input_target(
        input_tensor=input_tensor.unsqueeze(1),
        target_tensor=target_tensor,
        alphabet=alphabet
    )


if __name__ == '__main__':
    visualize_for_test()
    # visualize_for_source_file(path=os.path.join(os.getcwd(), 'data_dir/linux_kernel_mini.txt'))
