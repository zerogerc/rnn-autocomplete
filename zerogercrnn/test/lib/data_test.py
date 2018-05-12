import numpy as np
import pytest

from zerogercrnn.lib.data import split_train_validation, get_shuffled_indexes, get_random_index, DataChunk, \
    DataChunksPool, DataBucket, BucketsBatch, DataReader, BatchedDataGenerator


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


def test_data_chunks_pool_no_shuffle():
    data_size = 10
    splits = data_size // 2
    pool = create_test_data_pool(data_size, splits=splits, split_coefficient=0.5, shuffle=False)

    # emit all splits of data
    for i in range(splits):
        pool.start_epoch()
        assert pool.get_chunk().id == 2 * i
        assert pool.get_chunk().id == 2 * i + 1
        assert pool.is_epoch_finished()

    # do not crash on finish data and start from begining
    pool.start_epoch()
    assert pool.get_chunk().id == 0
    assert pool.get_chunk().id == 1


def test_data_chunks_pool_shuffle():
    data_size = 10
    splits = data_size // 2
    pool = create_test_data_pool(data_size, splits=splits, split_coefficient=0.5, shuffle=True)

    ids = set()
    first_chunk = None
    # emit all splits of data
    for i in range(splits):
        pool.start_epoch()
        if first_chunk is None:
            first_chunk = pool.get_chunk().id
            ids.add(first_chunk)
        else:
            ids.add(pool.get_chunk().id)
        ids.add(pool.get_chunk().id)
        assert pool.is_epoch_finished()

    assert len(ids) == data_size
    pool.start_epoch()


def test_data_chunks_pool_return_none_on_finished_epoch():
    data_size = 10
    pool = create_test_data_pool(data_size, splits=data_size // 2, split_coefficient=0.5, shuffle=True)

    assert pool.get_chunk() is None


def test_data_chunks_pool_exceptions_on_not_finished_epoch():
    data_size = 10
    pool = create_test_data_pool(data_size, splits=data_size // 2, split_coefficient=0.5, shuffle=True)

    with pytest.raises(Exception):
        pool.start_epoch()
        pool.start_epoch()


def test_data_bucket_emit_all_data_and_then_raise():
    data_size = 100
    seq_len = 50
    pool = create_test_data_pool(data_size, splits=10, split_coefficient=0.5, shuffle=False)
    pool.start_epoch()

    bucket = DataBucket(pool, seq_len)
    bucket.refill_if_necessary()
    for i in range(20):
        index, chunk = bucket.get_next_index_with_chunk()
        assert chunk.id == i // 2
        assert index == 50 * (i % 2)

    with pytest.raises(Exception) as excinfo:
        bucket.get_next_index_with_chunk()
    assert 'No data in bucket' in str(excinfo.value)


def test_buckets_batch():
    data_size = 60
    seq_len = 50
    batch_size = 3
    pool = create_test_data_pool(data_size, splits=10, split_coefficient=0.5, shuffle=False)
    pool.chunks[0].data = np.arange(200)

    cur_chunk = [0]
    chunk_numbers = [0, 1, 2, 0, 1, 2, 0, 3, 4, 0, 3, 4]

    def retriever(buckets):
        assert len(buckets) == batch_size
        for i in range(batch_size):
            index, chunk = buckets[i].get_next_index_with_chunk()
            assert chunk.id == chunk_numbers[cur_chunk[0]]
            cur_chunk[0] += 1

    batch = BucketsBatch(pool, seq_len=seq_len, batch_size=batch_size)
    for i, (data, forget_vector) in enumerate(batch.get_epoch(retriever)):
        if i == 0:
            assert np.all(forget_vector == 0)
        elif i == 1:
            assert np.all(forget_vector == 1)
        elif i == 2:
            assert np.all(forget_vector.view(-1).numpy().astype(int) == [1, 0, 0])
        elif i == 3:
            assert np.all(forget_vector == 1)


def test_batched_generator():
    data_size = 36
    seq_len = 50
    batch_size = 3
    reader = create_test_data_reader(data_size, data_size // 6, split_coefficient=5/6)

    def get_retriever(to_check_key, start=0):
        cur_chunk = [0]
        chunk_numbers = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]) + start

        def retriever(key, buckets):
            assert key == to_check_key
            for i in range(batch_size):
                index, chunk = buckets[i].get_next_index_with_chunk()
                assert chunk.id == chunk_numbers[cur_chunk[0]]
                cur_chunk[0] += 1

        return retriever

    generator = TestBatchedGenerator(reader, seq_len, batch_size, get_retriever('train'))
    for data in generator.get_train_generator():
        pass
        assert data[1].size()[0] == batch_size

    generator = TestBatchedGenerator(reader, seq_len, batch_size, get_retriever('validation', start=30))
    for data in generator.get_validation_generator():
        pass
        assert data[1].size()[0] == batch_size

    generator = TestBatchedGenerator(reader, seq_len, batch_size, get_retriever('eval'))
    for data in generator.get_eval_generator():
        pass
        assert data[1].size()[0] == batch_size


# region Utils

class TestBatchedGenerator(BatchedDataGenerator):

    def __init__(self, data_reader, seq_len, batch_size, retriever):
        super().__init__(data_reader, seq_len, batch_size, shuffle=False)
        self.retriever = retriever

    def _retrieve_batch(self, key, buckets):
        self.retriever(key, buckets)


def create_test_data_pool(data_size, splits, split_coefficient=0.5, shuffle=False):
    np.random.seed(1)
    reader = create_test_data_reader(2 * data_size, data_size, split_coefficient=split_coefficient)
    return DataChunksPool(reader.train_data, splits=splits, shuffle=shuffle)


def create_test_data_reader(train_length, test_length, split_coefficient=0.5):
    train_data = None
    test_data = None
    if train_length is not None:
        train_data = [create_test_data_chunk(100, i) for i in range(train_length)]
    if test_length is not None:
        test_data = [create_test_data_chunk(100, i) for i in range(test_length)]

    reader = DataReader()
    reader.train_data, reader.validation_data = split_train_validation(train_data, split_coefficient)
    reader.eval_data = test_data
    return reader


class TestDataChunk(DataChunk):
    def __init__(self, data, id):
        self.id = id
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


def create_test_data_chunk(length, _id):
    return TestDataChunk(np.arange(length), _id)

# endregion
