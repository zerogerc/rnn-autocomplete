import numpy as np

from zerogercrnn.lib.data import split_train_validation, get_shuffled_indexes, get_random_index, DataChunk, DataReader


def test_split_train_validation():
    data = np.arange(0, 10)
    train, val = split_train_validation(data, 0.8)
    assert np.all(train == np.arange(0, 8))
    assert np.all(val == np.array([8, 9]))


def test_get_shuffled_indexes():
    data = get_shuffled_indexes(100)
    assert len(data) == 100


def test_get_random_index():
    np.random.seed(1)
    for i in range(100):
        assert get_random_index(100) < 100


def test_data_chunks_pool():
    pass


def create_test_data_reader(train_length, test_length):
    train_data = None
    test_data = None
    if train_length is not None:
        train_data = [create_test_data_chunk(100) for _ in range(train_length)]
    if test_length is not None:
        test_data = [create_test_data_chunk(100) for _ in range(test_length)]

    reader = DataReader()
    reader.train_data, reader.validation_data = split_train_validation(train_data, 0.5)
    reader.eval_data = test_data
    return reader


class TestDataChunk(DataChunk):
    def __init__(self, data):
        self.data = data
        self.seq_len = None

    def prepare_data(self, seq_len):
        self.data = self.data[:len(self.data) - (len(self.data) % seq_len)]

    def get_by_index(self, index):
        assert self.seq_len is not None
        assert len(self.data) % self.seq_len == 0
        assert (index < self.size())
        return self.data[index]

    def size(self):
        return len(self.data)


def create_test_data_chunk(length):
    return TestDataChunk(np.arange(length))
