import numpy as np
from typing import Dict

from zerogercrnn.lib.data.general import DataGenerator


class BatcherDataGenerator(DataGenerator):
    """Encapsulates batcher architecture into general interface for data."""

    def __init__(self, batcher, batch_size):
        self.batcher = batcher
        self.batch_size = batch_size

    def get_train_generator(self):
        return self.batcher.data_map['train'].get_batched_epoch(self.batch_size)

    def get_validation_generator(self):
        return self.batcher.data_map['validation'].get_batched_epoch(self.batch_size)


class BatchNode:
    def __init__(self, key, size, picker):
        self.key = key
        self.size = size

        self.picker = picker

        self.indexes = None
        self.cur_id = 0

    def get_batched_random(self, batch_size):
        """Infinite generator of random batches."""
        while True:
            yield self.picker(self.__get_batch_indexes__(batch_size))

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
            yield self.picker(cur)

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
        self.data_map = {}

    def add_data(self, key, size, picker):
        """
        :param key: key to store batcher in map.
        :param size: size of the dataset. Used to produce indexes. for batch
        :param picker: function that returns (input_tensor, target_tensor) by given indexes.
        """
        self.data_map[key] = BatchNode(
            key=key,
            size=size,
            picker=picker
        )


if __name__ == '__main__':
    pass
    # x = torch.zeros((2, 2))
    # print(x.size())
    # print(x.unsqueeze(1).size())
    # print(general_data_picker(torch.eye(10), 3, np.array([1, 3]))[:, 0, :])
