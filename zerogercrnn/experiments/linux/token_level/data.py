import os

import numpy as np
import torch

from zerogercrnn.experiments.linux.batcher import Batcher
from zerogercrnn.lib.data.token_level import Corpus


class DataPicker:
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def add_data(self, batcher, key, tensor, ntokens):
        """Adds data to the batcher that will transform data from row tensor into input and target on the fly.
        Data is divied on the chunks of len seq_len."""
        sz = (tensor.size()[0] - 1) // self.seq_len  # divide all data by seq_len chunks
        batcher.add_data(
            key=key,
            size=sz,
            picker=lambda indexes: self.pick_input_and_target_batch(tensor, indexes, ntokens))

    def pick_input_and_target_batch(self, tensor, indexes, ntokens):
        """Picks input and target batches with given index of starts of batches."""

        input_tensor = self.pick_batch(
            tensor,
            indexes,
            lambda data, start: self.pick_single_input(data, start, ntokens)
        )
        target_tensor = self.pick_batch(
            tensor,
            indexes,
            lambda data, start: self.pick_single_target(data, start)
        )

        return input_tensor, target_tensor

    def pick_batch(self, data_tensor, start_indexes, single_picker):
        """Pick batch of data from tensor using start indexes of chunks."""

        # call chunk_pick and add one dimension for batching
        out = [single_picker(data_tensor, start * self.seq_len).unsqueeze(1) for start in start_indexes]
        return torch.cat(out, dim=1)

    def pick_single_input(self, data_tensor, start, ntokens):  # one hot encoding of data
        """Picks input from the tensor with row data."""
        return data_tensor[np.arange(start, start + self.seq_len), :]
        # """Picks input from the tensor with row data by convertin it to the one hot representation."""
        # positions = data_tensor[np.arange(start, start + self.seq_len), :]
        # one_hot = torch.zeros(self.seq_len, ntokens)
        # one_hot.scatter_(1, positions.view(self.seq_len, 1), 1.)
        # return one_hot

    def pick_single_target(self, data_tensor, start):  # index of target (input shifted by one)
        """Picks target from the tensor with row data."""
        return data_tensor[np.arange(start + 1, start + 1 + self.seq_len), :]


def read_data(datadir, tokens_path, *, seq_len=100):
    """Supposes that train/validation/test datasets is stored in the path
    with names train.txt/validation.txt/test.txt.

    :param path: path to the data directory with train.txt/validation.txt/test.txt files.
    :return: batcher with train/validation/test and Corpus.
    """

    corpus = Corpus.create_from_data_dir(datadir, tokens_path=tokens_path)
    batcher = Batcher()
    ntokens = len(corpus.tokens)

    picker = DataPicker(seq_len=seq_len)
    picker.add_data(batcher, 'train', corpus.train.unsqueeze(1), ntokens)
    picker.add_data(batcher, 'validation', corpus.valid.unsqueeze(1), ntokens)
    picker.add_data(batcher, 'test', corpus.test.unsqueeze(1), ntokens)
    return batcher, corpus


def read_data_mini(single, tokens_path, *, seq_len):
    """
    :param single: path to the data file.
    :return: batcher with train/validation/test and Corpus.
    """
    corpus = Corpus.create_from_single_file(path=single, tokens_path=tokens_path)
    batcher = Batcher()
    ntokens = len(corpus.tokens)

    picker = DataPicker(seq_len=seq_len)
    picker.add_data(batcher, 'train', corpus.train, ntokens)
    picker.add_data(batcher, 'validation', corpus.valid, ntokens)
    picker.add_data(batcher, 'test', corpus.test, ntokens)
    return batcher, corpus


if __name__ == '__main__':
    pass
    # path = os.path.join(ROOT_DIR, 'data/kernel_concatenated/test.txt')
    # f = open(path, 'r', encoding=DEFAULT_ENCODING)
    # print(len(f.read()))
    # batcher, corpus = read_data_mini()
    # i, t = next(batcher.data_map['train'].get_batched_random(batch_size=8))
    # print(i.size())
    # print(t.size())
