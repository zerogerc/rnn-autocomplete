import numpy as np
import torch
from torch.autograd import Variable

from typing import Dict


class BatchNode:
    def __init__(self, key, size, input_picker, output_picker):
        self.key = key
        self.size = size

        self.input_picker = input_picker
        self.output_picker = output_picker

        self.indexes = None
        self.cur_id = 0

    def get_batched_random(self, batch_size):
        """Infinite generator of random batches."""
        while True:
            indexes = self.__get_batch_indexes__(batch_size)
            input_tensor = Variable(self.input_picker(indexes))
            target_tensor = Variable(self.output_picker(indexes))
            yield input_tensor, target_tensor

    def get_batched_epoch(self, batch_size):
        """Returns generator for batched input. It will run through all inputs except 
        inputs that are not in any batch (residue). 
        
        :param batch_size: size of batch to split inputs.
        :return: generator through batches that contains all inputs (except residue).
        """
        indexes = np.arange(0, self.size)
        np.random.shuffle(indexes)
        cur_id = 0
        while cur_id + batch_size < self.size:
            cur = indexes[cur_id: cur_id + batch_size]
            cur_id += batch_size
            input_tensor = Variable(self.input_picker(cur))
            target_tensor = Variable(self.output_picker(cur))
            yield input_tensor, target_tensor

    def __get_batch_indexes__(self, batch_size):
        if self.indexes is None:
            self.indexes = np.arange(0, self.size)
            np.random.shuffle(self.indexes)

        if self.cur_id + batch_size > self.size:
            np.random.shuffle(self.indexes)
            self.cur_id = 0

        self.cur_id += batch_size
        return self.indexes[self.cur_id - batch_size: self.cur_id]


class Batcher:
    def __init__(self):
        self.data_map: Dict[str, BatchNode] = {}

    def add_rnn_data(self, key, input_tensor, target_tensor):
        """Store batcher by key. You could get generator of batched data using get_batched.
        
        :param key: name to store batcher in map.
        :param input_tensor: tensor in the form (seq_len, data_len, input_size).
        :param target_tensor: tensor in the form (seq_len, data_len).
        """
        self.data_map[key] = BatchNode(
            key,
            input_tensor.size()[1],
            lambda indexes: input_tensor[:, indexes, :],
            lambda indexes: target_tensor[:, indexes]
        )

    def add_rnn_data_generator(self, key, data_tensor, seq_len, input_chunk_picker=None, target_chunk_picker=None):
        """Store batcher by key. You could get generator of batched data using get_batched.
        
        This batcher will generate batch from input sequence.
        Target will be input sequence shifted by 1.
        
        :param key: name to store batcher in map.
        :param data_tensor: data tensor of input sequence (e.g. text) in the form (data_len, input_size).
        :param seq_len: size of window for one input.
        :param input_chunk_picker: picker for chunks of data from data_tensor to input.
        :param target_chunk_picker: picker for chunks if data from data_tensor to target.
        """
        self.data_map[key] = BatchNode(
            key,
            data_tensor.size()[0] - seq_len,
            lambda indexes: rnn_data_picker(data_tensor, seq_len, indexes, chunk_picker=input_chunk_picker),
            lambda indexes: rnn_data_picker(data_tensor, seq_len, indexes + 1, chunk_picker=target_chunk_picker)
            # shifted by 1
        )


class ChunkPicker:
    """
    Base class for data pickers
    """

    def pick(self, data_tensor, start, chunk_len):
        pass


class ChunkIdPicker(ChunkPicker):
    def __init__(self):
        super(ChunkIdPicker, self).__init__()

    def pick(self, data_tensor, start, chunk_len):
        return data_tensor[np.arange(start, start + chunk_len), :]


class ChunkOneHotPicker(ChunkPicker):
    def __init__(self, ntokens):
        super(ChunkOneHotPicker, self).__init__()
        self.ntokens = ntokens

    def pick(self, data_tensor, start, chunk_len):
        positions = data_tensor[np.arange(start, start + chunk_len), :]
        one_hot = torch.zeros(chunk_len, self.ntokens)
        one_hot.scatter_(1, positions.view(chunk_len, 1), 1.)
        return one_hot


def rnn_data_picker(data_tensor, chunk_len, start_indexes, chunk_picker=None):
    """Choose subsequent chunks of data from 2d tensor.
    
    :param data_tensor: 2d tensor
    :param chunk_len: chunk length
    :param start_indexes: indexes of chunk starts
    :param chunk_picker: function that pick subsequent chunk of data from tensor.
    :return: tensor of size [chunk_len, len(start_indexes), input_size]
    """
    chunk_picker = ChunkIdPicker() if chunk_picker is None else chunk_picker

    # call chunk_pick and add one dimension for batching
    out = [chunk_picker.pick(data_tensor, start, chunk_len).unsqueeze(1) for start in start_indexes]
    return torch.cat(out, dim=1)


if __name__ == '__main__':
    x = torch.zeros((2, 2))
    print(x.size())
    print(x.unsqueeze(1).size())
    # print(general_data_picker(torch.eye(10), 3, np.array([1, 3]))[:, 0, :])
