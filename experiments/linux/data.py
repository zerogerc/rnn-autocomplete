import os
import numpy as np
import torch

from global_constants import ROOT_DIR, DEFAULT_ENCODING
from experiments.linux.constants import alphabet
from lib.utils.split import split_data
from lib.data.character import Corpus

from experiments.linux.batcher import Batcher

SEQ_LEN = 100
BATCH_SIZE = 100

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


def add_data(batcher, key, tensor, ntokens):
    sz = (tensor.size()[0] - 1) // SEQ_LEN  # divide all data by SEQ_LEN chunks

    def picker(indexes):
        input_tensor = pick_batch(
            tensor,
            indexes,
            lambda data, start: pick_single_input(data, start, ntokens)
        )
        target_tensor = pick_batch(
            tensor,
            indexes,
            pick_single_target
        )

        return input_tensor, target_tensor

    batcher.add_data(key=key, size=sz, picker=picker)


def read_data():
    corpus = Corpus(os.path.join(ROOT_DIR, 'data/kernel_concatenated'), letters=alphabet)

    batcher = Batcher()

    ntokens = len(corpus.all_letters)
    add_data(batcher, 'train', corpus.train.unsqueeze(1), ntokens)
    add_data(batcher, 'validation', corpus.valid.unsqueeze(1), ntokens)
    add_data(batcher, 'test', corpus.test.unsqueeze(1), ntokens)
    return batcher, corpus


def read_data_mini():
    corpus = Corpus(os.path.join(ROOT_DIR, 'data'), single_file='linux_kernel_mini.txt', letters=alphabet)
    input_tensor = corpus.single.unsqueeze(1)

    train, validation, test = split_data(input_tensor, validation_percentage=0.1, test_percentage=0.1, shuffle=False)
    batcher = Batcher()

    ntokens = len(corpus.all_letters)
    add_data(batcher, 'train', train, ntokens)
    add_data(batcher, 'validation', validation, ntokens)
    add_data(batcher, 'test', test, ntokens)
    return batcher, corpus

if __name__ == '__main__':
    path = os.path.join(ROOT_DIR, 'data/kernel_concatenated/test.txt')
    f = open(path, 'r', encoding=DEFAULT_ENCODING)
    print(len(f.read()))
    # batcher, corpus = read_data_mini()
    # i, t = next(batcher.data_map['train'].get_batched_random(batch_size=8))
    # print(i.size())
    # print(t.size())