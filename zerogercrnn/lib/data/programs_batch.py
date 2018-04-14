from abc import abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from zerogercrnn.lib.data.general import DataGenerator


def split_train_validation(data, split_coefficient):
    train_examples = int(len(data) * split_coefficient)
    return data[:train_examples], data[train_examples:len(data)]


def get_shuffled_indexes(length):
    temp = np.arange(length)
    np.random.shuffle(temp)
    return temp


def get_random_index(length):
    return np.random.randint(length)


class DataChunk:

    @abstractmethod
    def prepare_data(self, seq_len):
        """Align data with seq_len."""
        pass

    @abstractmethod
    def get_by_index(self, index, additional):
        pass

    @abstractmethod
    def size(self):
        pass


class BatchedDataGenerator(DataGenerator):
    """Provides batched data for training and evaluation of model."""

    def __init__(self, data_reader, seq_len, batch_size, cuda):
        super(BatchedDataGenerator, self).__init__()

        self.data_reader = data_reader
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cuda = cuda

        if data_reader.train_data is not None:
            self.data_train = self._prepare_data_(data_reader.train_data)

        if data_reader.validation_data is not None:
            self.data_validation = self._prepare_data_(data_reader.validation_data)

        if data_reader.eval_data is not None:
            self.data_eval = self._prepare_data_(data_reader.eval_data)

        # Share indexes between epochs because we want one epoch to be 1/5 of dataset
        # Map is for storing train/validation separately
        self.datasets = {
            'train': self.data_train,
            'validation': self.data_validation,
            'eval': self.data_eval
        }
        self.indexes = {}
        self.current = {}
        self.right = {}
        self.forget_vector = {}

        self.current_key = 'train'
        self.epoch_finished = False

        def getter():
            indexes = self.indexes[self.current_key]
            current = self.current[self.current_key]
            dataset = self.datasets[self.current_key]
            right = self.right[self.current_key]

            if current == right:
                self.epoch_finished = True
                return None

            chunk = dataset[indexes[current]]
            self.current[self.current_key] = current + 1

            return chunk

        self.buckets = []
        for i in range(self.batch_size):
            self.buckets.append(
                DataBucket(
                    seq_len=self.seq_len,
                    cuda=self.cuda,
                    getter=getter
                ))

    @abstractmethod
    def _retrieve_batch_(self, key):
        """Here you could suppose that you have non-empty buckets and you could extract data."""
        pass

    # override
    def get_train_generator(self):
        return self._get_batched_epoch_(dataset=self.data_train, key='train')

    # override
    def get_validation_generator(self):
        return self._get_batched_epoch_(dataset=self.data_validation, key='validation')

    # override
    def get_eval_generator(self):
        return self._get_batched_epoch_(dataset=self.data_eval, key='eval')

    def _get_batched_epoch_(self, dataset, key):
        """Returns generator over batched data of all files in the dataset."""

        self.current_key = key
        self.epoch_finished = False

        # Share indexes between epochs because we want one epoch to be 1/5 of dataset
        if key not in self.indexes:
            self._init_epoch_state_(key, data_len=len(dataset))

        for b in self.buckets:
            if b.is_empty():
                b.try_refill()

        while True:
            current = self.current[self.current_key]
            if current % 1000 == 0:
                print('Processed {} programs'.format(current))

            if not self.epoch_finished:
                yield self._retrieve_batch_(key), self.forget_vector[key]
            else:
                break

        self.right[key] = min(self.current[key] + len(self.datasets[key]) // 5, len(self.datasets))

        if current >= len(self.indexes[self.current_key]):
            self._reset_epoch_state_(key)

    def _prepare_data_(self, data):
        for i in tqdm(range(len(data))):
            data[i].prepare_data(self.seq_len)

        return data

    def _init_epoch_state_(self, key, data_len):
        self.indexes[key] = get_shuffled_indexes(data_len)
        self.current[key] = 0
        self.right[key] = len(self.datasets[key]) // 5
        self.forget_vector[key] = torch.ones(self.batch_size, 1)

        if self.cuda:
            self.forget_vector[key] = self.forget_vector[key].cuda()

    def _reset_epoch_state_(self, key):
        self.indexes.pop(key)
        self.current.pop(key)
        self.forget_vector.pop(key)
        self.right.pop(key)


class DataBucket:
    """Bucket with DataChunks."""

    def __init__(self, seq_len, cuda, getter):
        self.seq_len = seq_len
        self.cuda = cuda
        self.source: DataChunk = None
        self.getter = getter
        self.index = 0

    def add_chunk(self, data_chunk: DataChunk):
        """Adds the whole source file to the bucket."""
        assert data_chunk.size() % self.seq_len == 0
        self.source = data_chunk
        self.index = 0

    def get_next_index(self):
        """Return input and target tensors from attached DataChunk with lenghts seq_len - 1."""
        if self.is_empty():
            print(self.source)
            print(self.index)
            raise Exception('No data in bucket')
        self.index += self.seq_len
        start = self.index - self.seq_len
        return start

    def try_refill(self):
        source = self.getter()
        if source is None:
            self.source = None
            self.index = 0
        else:
            self.add_chunk(source)

    def is_empty(self):
        """Indicates whether this bucket contains at least one more sequence."""
        return (self.source is None) or (self.source.size() == self.index)

    def clear(self):
        """Remove attached SourceFile from this bucket."""
        self.source = None
        self.index = 0
