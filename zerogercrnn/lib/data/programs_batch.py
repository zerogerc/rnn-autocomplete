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
    def get_by_index(self, index):
        pass

    @abstractmethod
    def size(self):
        pass


class BatchedDataGenerator(DataGenerator):
    """Provides batched data for training and evaluation of model."""

    def __init__(self, data_reader, seq_len, batch_size):
        super(BatchedDataGenerator, self).__init__()

        self.data_reader = data_reader
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.cuda = self.data_reader.cuda

        if data_reader.train_data is not None:
            self.data_train = self._prepare_data_(data_reader.train_data)

        if data_reader.validation_data is not None:
            self.data_validation = self._prepare_data_(data_reader.validation_data)

        if data_reader.eval_data is not None:
            self.data_eval = self._prepare_data_(data_reader.eval_data)

        # Share indexes between epochs because we want one epoch to be 1/5 of dataset
        # Map is for storing train/validation separately
        self.indexes = {}
        self.current = {}
        self.forget_vector = {}

        self.buckets = []
        for i in range(self.batch_size):
            self.buckets.append(DataBucket(seq_len=self.seq_len))

    @abstractmethod
    def _retrieve_batch_(self):
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

        # Share indexes between epochs because we want one epoch to be 1/5 of dataset
        if key not in self.indexes:
            self._init_epoch_state_(key, data_len=len(dataset))

        indexes = self.indexes[key]
        current = self.current[key]

        # Parse programs till this number
        right = min(current + len(dataset) // 5, len(dataset))

        while True:
            cont = True  # indicates if we need to continue add new chunks or finish epoch

            # Refill empty buckets.
            for bn in range(len(self.buckets)):
                bucket = self.buckets[bn]

                if bucket.is_empty():
                    if current == right:
                        cont = False
                        break

                    # Chunk changed need to forget hidden state
                    self.forget_vector[key][bn][0] = 0.

                    bucket.add_chunk(data_chunk=dataset[indexes[current]])
                    current += 1
                else:
                    # Pass states to the next iteration (i.e. hidden state)
                    self.forget_vector[key][bn][0] = 1.

            if cont:
                yield self._retrieve_batch_(), self.forget_vector[key]
            else:
                break

            if current == right:
                break

        self.indexes[key] = indexes
        self.current[key] = current

        if current >= len(indexes):
            self._reset_epoch_state_(key)

    def _prepare_data_(self, data):
        for i in tqdm(range(len(data))):
            data[i].prepare_data(self.seq_len)

        return data

    def _init_epoch_state_(self, key, data_len):
        self.indexes[key] = get_shuffled_indexes(data_len)
        self.current[key] = 0
        self.forget_vector[key] = torch.ones(self.batch_size, 1)

        if self.cuda:
            self.forget_vector[key] = self.forget_vector[key].cuda()

    def _reset_epoch_state_(self, key):
        self.indexes.pop(key)
        self.current.pop(key)
        self.forget_vector.pop(key)


class DataBucket:
    """Bucket with DataChunks."""

    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.source: DataChunk = None
        self.index = 0

    def add_chunk(self, data_chunk: DataChunk):
        """Adds the whole source file to the bucket."""
        assert data_chunk.size() % self.seq_len == 0
        self.source = data_chunk
        self.index = 0

    def get_next_seq(self):
        """Return DataChunk with len equal to seq_len
        """
        assert not self.is_empty()
        self.index += self.seq_len
        start = self.index - self.seq_len
        return self.source.get_by_index(start)

    def is_empty(self):
        """Indicates whether this bucket contains at least one more sequence."""
        return (self.source is None) or (self.source.size() == self.index)

    def clear(self):
        """Remove attached SourceFile from this bucket."""
        self.source = None
        self.index = 0
