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

        self.indexes = np.arange(0, self.size)
        np.random.shuffle(self.indexes)
        self.cur_id = 0

    def get_batch_indexes(self, batch_size):
        if self.cur_id + batch_size > self.size:
            np.random.shuffle(self.indexes)
            self.cur_id = 0

        return self.indexes[self.cur_id: self.cur_id + batch_size]

    def get_batched(self, batch_size):
        while True:
            indexes = self.get_batch_indexes(batch_size)
            input_tensor = Variable(self.input_picker(indexes))
            target_tensor = Variable(self.output_picker(indexes))
            yield input_tensor, target_tensor


class Batcher:
    def __init__(self):
        self.data_map: Dict[str, BatchNode] = {}

    def add_rnn_data(self, key, input_tensor, target_tensor):
        """Store batcher by key. You could get generator of batched data using get_batched.
        
        :param key: name to store batcher in map
        :param input_tensor: tensor in the form (seq_len, data_len, input_size)
        :param target_tensor: tensor in the form (seq_len, data_len)
        """
        self.data_map[key] = BatchNode(
            key,
            input_tensor.size()[1],
            lambda indexes: input_tensor[:, indexes, :],
            lambda indexes: target_tensor[:, indexes]
        )

    def add_rnn_data_generator(self, key, data_tensor, seq_len):
        """Store batcher by key. You could get generator of batched data using get_batched.
        
        This batcher will generate batch from input sequence.
        Target will be input sequence shifted by 1.
        
        :param key: name to store batcher in map
        :param data_tensor: data tensor of input sequence (e.g. text) in the form (data_len, input_size)
        :param seq_len: size of window for one input
        """
        self.data_map[key] = BatchNode(
            key,
            data_tensor.size()[0] - seq_len,
            lambda indexes: choose_from_data(data_tensor, seq_len, indexes),
            lambda indexes: choose_from_data(data_tensor, seq_len, indexes + 1) # shifted by 1
        )


def choose_from_data(data_tensor, chunk_len, start_indexes):
    """Choose subsequent chunks of data from 2d tensor.
    
    :param data_tensor: 2d tensor
    :param chunk_len: chunk length
    :param start_indexes: indexes of chunk starts
    :return: tensor of size [chunk_len, len(start_indexes), input_size]
    """
    input_len = data_tensor.size()[1]

    out = []
    for start in start_indexes:
        out.append(data_tensor[np.arange(start, start + chunk_len), :].view(chunk_len, 1, input_len))
    return torch.cat(out, dim=1)

if __name__ == '__main__':
    print(choose_from_data(torch.eye(10), 3, np.array([1,3]))[:, 0, :])