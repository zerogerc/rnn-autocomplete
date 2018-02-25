import os

import numpy as np
import torch
from torch.autograd import Variable

from zerogercrnn.experiments.linux.token_level.main import create_gru
from zerogercrnn.experiments.linux.token_level.data import read_data_mini
from zerogercrnn.experiments.linux.token_level.gru_model import GRULinuxNetwork
from zerogercrnn.experiments.linux.constants import HOME_DIR
from zerogercrnn.lib.utils.state import load_if_saved
from zerogercrnn.lib.visualization.text import show_token_diff

SEQ_LEN = 10000

def pick_single_input(data_tensor, start, ntokens):
    return data_tensor[np.arange(start, start + SEQ_LEN), :]


def pick_single_target(data_tensor, start):  # index of target (input shifted by one)
    return data_tensor[np.arange(start + 1, start + 1 + SEQ_LEN), :]


def pick_batch(data_tensor, start_indexes, single_picker):
    # call chunk_pick and add one dimension for batching
    out = [single_picker(data_tensor, start * SEQ_LEN).unsqueeze(1) for start in start_indexes]
    return torch.cat(out, dim=1)


def visualize_for_input_target(input_tensor, target_tensor, tokens):
    network = create_gru(len(tokens))
    load_if_saved(network, os.path.join(os.getcwd(), 'saved_models/model_epoch_6'))

    output = network(Variable(input_tensor))

    predicted = convert(torch.max(output.squeeze(1), 1)[1].data, tokens=tokens)
    actual = convert(target_tensor.view(-1), tokens=tokens)

    show_token_diff(predicted=predicted, actual=actual, file=os.path.join(os.getcwd(), 'result.html'))


def convert(token_positions, tokens):
    tokenized = []
    for lp in token_positions:
        tokenized.append(tokens[lp])
    return tokenized


def visualize_for_test():
    batcher, corpus = read_data_mini(
        single=os.path.join(HOME_DIR, 'data_dir/linux_kernel_mini.txt'),
        tokens_path=os.path.join(os.getcwd(), 'tokens.txt'),
        seq_len=SEQ_LEN
    )

    # batcher, corpus = read_data(
    #     datadir=os.path.join(os.getcwd(), 'data_dir/kernel_concat'),
    #     seq_len=SEQ_LEN
    # )

    data_generator = batcher.data_map['test'].get_batched_random(batch_size=1)

    input_tensor, target_tensor = next(data_generator)
    visualize_for_input_target(input_tensor, target_tensor, tokens=corpus.tokens)


# def visualize_for_source_file(path):
#     data = tokenize(path=path, alphabet=alphabet).unsqueeze(1)  # tensor of size [Nx1]
#
#     input_tensor = pick_single_input(data_tensor=data, start=0, ntokens=len(alphabet))
#     target_tensor = pick_single_target(data_tensor=data, start=0)
#
#     visualize_for_input_target(
#         input_tensor=input_tensor.unsqueeze(1),
#         target_tensor=target_tensor,
#         tokens=tokens
#     )


if __name__ == '__main__':
    visualize_for_test()
    # visualize_for_source_file(path=os.path.join(os.getcwd(), 'data_dir/linux_kernel_mini.txt'))
