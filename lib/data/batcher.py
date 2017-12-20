import numpy as np
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
        """
        Store batcher by key. You could get generator of batched data using get_batched.
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
