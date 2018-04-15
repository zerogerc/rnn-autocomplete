import json

import torch
from tqdm import tqdm

from zerogercrnn.lib.data.general import DataReader
from zerogercrnn.lib.data.programs_batch import DataChunk, BatchedDataGenerator, split_train_validation

# hack for tqdm
tqdm.monitor_interval = 0

from zerogercrnn.lib.embedding import Embeddings

VECTOR_FILE = 'data/tokens/vectors.txt'
TRAIN_FILE = 'data/tokens/file_train.json'
EVAL_FILE = 'data/tokens/file_eval.json'
ENCODING = 'ISO-8859-1'


class TokensDataChunk(DataChunk):
    """Wrapper on tensor of size [program_len, embedding_size]."""

    def __init__(self, one_hot_tensor, embeddings: Embeddings, cuda):
        super().__init__()

        self.embeddings = embeddings
        self.embeddings_buffer = None
        self.embeddings_cache = None
        self.one_hot_tensor = one_hot_tensor
        self.seq_len = None
        self.cuda = cuda

    def prepare_data(self, seq_len):
        self.seq_len = seq_len
        ln = self.size() - self.size() % seq_len
        self.one_hot_tensor = self.one_hot_tensor.narrow(dimension=0, start=0, length=ln)

    def set_embeddings_buffer(self, buffer):
        self.embeddings_buffer = buffer

    def init_cache(self):
        self.one_hot_tensor = self.one_hot_tensor.narrow(
            dimension=0,
            start=0,
            length=min(self.one_hot_tensor.size()[0], 20 * self.seq_len)
        )
        self.embeddings_cache = self.embeddings.index_select(self.one_hot_tensor)

        if self.cuda:
            self.one_hot_tensor.cuda()
            self.embeddings_cache.cuda()

    def drop_cache(self):
        self.embeddings_cache = None

    def get_by_index(self, index):
        if self.seq_len is None:
            raise Exception('You should call prepare_data with specified seq_len first')
        if index + self.seq_len > self.size():
            raise Exception('Not enough data in chunk')
        if self.embeddings_buffer is None:
            raise Exception('You should set embeddings buffer first')

        if index == 0:
            self.init_cache()

        input_tensor_emb = self.embeddings_cache.narrow(dimension=0, start=index, length=self.seq_len - 1)
        target_tensor = self.one_hot_tensor.narrow(dimension=0, start=index + 1, length=self.seq_len - 1)

        if index + self.seq_len + self.seq_len > self.size():
            self.drop_cache()

        return input_tensor_emb, target_tensor

    def size(self):
        return self.one_hot_tensor.size()[0]


class TokensDataGenerator(BatchedDataGenerator):

    def __init__(self, data_reader: DataReader, seq_len, batch_size, embeddings_size, cuda):
        super().__init__(data_reader, seq_len=seq_len, batch_size=batch_size, cuda=cuda)

        self.embeddings_size = embeddings_size
        self.embeddings_buffer = {}

    def _retrieve_batch(self, key, buckets):
        inputs = []
        targets = []

        self._init_buffer_if_necessary(key)

        for b in buckets:
            id, chunk = b.get_next_index_with_chunk()

            chunk.set_embeddings_buffer(self.embeddings_buffer[key])
            i, t = chunk.get_by_index(id)

            inputs.append(i)
            targets.append(t)

        return torch.stack(inputs, dim=1), torch.stack(targets, dim=1)

    def _init_buffer_if_necessary(self, key):
        if key not in self.embeddings_buffer.keys():
            self.embeddings_buffer[key] = torch.FloatTensor(self.seq_len - 1, self.batch_size, self.embeddings_size)
            if self.cuda:
                self.embeddings_buffer[key] = self.embeddings_buffer[key].pin_memory()
                # self.embeddings_buffer[key] = self.embeddings_buffer[key].cuda()


class MockDataReader:
    """Fast analog of DataReader for testing."""

    def __init__(self, cuda=True):
        self.cuda = cuda and torch.cuda.is_available()

        e_t = torch.randn((100, 50))
        o_t = torch.ones(100)

        if self.cuda:
            e_t.cuda()
            o_t.cude()

        self.data_train = [TokensDataChunk(e_t, o_t) for i in range(400)]
        self.data_validation = [TokensDataChunk(e_t, o_t) for i in range(400)]
        self.data_eval = [TokensDataChunk(e_t, o_t) for i in range(400)]


class TokensDataReader(DataReader):
    """Reads the data from file and transform it to torch Tensors."""

    def __init__(self, train_file, eval_file, embeddings: Embeddings, seq_len, cuda, limit=100000):
        super().__init__()
        self.train_file = train_file
        self.eval_file = eval_file
        self.embeddings = embeddings
        self.seq_len = seq_len
        self.cuda = cuda

        if self.cuda:
            # self.embeddings.pin_memory()
            self.embeddings.cuda()

        print('Start data reading')
        if self.train_file is not None:
            self.train_data, self.validation_data = split_train_validation(
                data=self._read_file(train_file, limit=limit, label='Train'),
                split_coefficient=0.8
            )

        if self.eval_file is not None:
            self.eval_data = self._read_file(eval_file, limit=limit, label='Eval')

        print('Data reading finished')
        print('Train size: {}, Validation size: {}, Eval size: {}'.format(
            len(self.train_data),
            len(self.validation_data),
            len(self.eval_data)
        ))

    def _read_file(self, file_path, limit=100000, label='Data'):
        print('Reading {} ... '.format(label))
        data = []
        it = 0
        for l in tqdm(open(file=file_path, mode='r', encoding=ENCODING), total=limit):
            it += 1

            tokens = json.loads(l)
            one_hot = torch.LongTensor(tokens)
            if self.cuda:
                # one_hot = one_hot.pin_memory()
                one_hot = one_hot.cuda()

            data.append(TokensDataChunk(
                one_hot_tensor=one_hot,
                embeddings=self.embeddings,
                cuda=self.cuda
            ))

            if (limit is not None) and (it == limit):
                break

        return list(filter(lambda d: d.size() >= self.seq_len, data))


if __name__ == '__main__':
    # x = torch.FloatTensor(5, 4)
    # ids = torch.LongTensor([0, 0, 1, 2, 1])
    #
    # print(torch.index_select(x, dim=0, index=ids))

    emb = Embeddings(embeddings_size=50, vector_file=VECTOR_FILE)
    x = 0
    # data_reader = TokensDataReader(
    #     train_file=TRAIN_FILE,
    #     eval_file=None,
    #     embeddings=emb,
    #     seq_len=10,
    #     cuda=False,
    #     limit=1000
    # )
    #
    # data_generator = TokensDataGenerator(
    #     data_reader=data_reader,
    #     seq_len=10,
    #     batch_size=10,
    #     cuda=False
    # )
    #
    # for iter_data in data_generator.get_train_generator():
    #     print(iter_data)
    #     break
