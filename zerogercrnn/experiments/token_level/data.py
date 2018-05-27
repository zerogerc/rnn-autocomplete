import json

import torch
from tqdm import tqdm

from zerogercrnn.lib.data import DataChunk, BatchedDataGenerator, split_train_validation, DataReader
from zerogercrnn.lib.utils import get_best_device

# hack for tqdm
tqdm.monitor_interval = 0

from zerogercrnn.lib.calculation import pad_tensor

VECTOR_FILE = 'data/tokens/vectors.txt'
TRAIN_FILE = 'data/tokens/file_train.json'
EVAL_FILE = 'data/tokens/file_eval.json'
ENCODING = 'ISO-8859-1'


class TokensDataChunk(DataChunk):
    def __init__(self, one_hot_tensor):
        super().__init__()

        self.one_hot_tensor = one_hot_tensor.to(get_best_device())
        self.seq_len = None

    def prepare_data(self, seq_len):
        self.seq_len = seq_len
        self.one_hot_tensor = pad_tensor(tensor=self.one_hot_tensor, seq_len=seq_len)

    def get_by_index(self, index):
        if self.seq_len is None:
            raise Exception('You should call prepare_data with specified seq_len first')
        if index + self.seq_len > self.size():
            raise Exception('Not enough data in chunk')

        input_tensor_emb = self.one_hot_tensor.narrow(dim=0, start=index, length=self.seq_len - 1)
        target_tensor = self.one_hot_tensor.narrow(dim=0, start=index + 1, length=self.seq_len - 1)

        return input_tensor_emb, target_tensor

    def size(self):
        return self.one_hot_tensor.size()[0]


class TokensDataReader(DataReader):
    """Reads the data from file and transform it to torch Tensors."""

    def __init__(self, train_file, eval_file, seq_len, limit=100000):
        super().__init__()
        self.train_file = train_file
        self.eval_file = eval_file
        self.seq_len = seq_len

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
            one_hot = torch.LongTensor(tokens).to(get_best_device())

            data.append(TokensDataChunk(one_hot_tensor=one_hot))

            if (limit is not None) and (it == limit):
                break

        return list(filter(lambda d: d.size() >= self.seq_len, data))


class TokensDataGenerator(BatchedDataGenerator):

    def __init__(self, data_reader: DataReader, seq_len, batch_size):
        super().__init__(data_reader, seq_len=seq_len, batch_size=batch_size)

    def _retrieve_batch(self, key, buckets):
        inputs = []
        targets = []

        for b in buckets:
            id, chunk = b.get_next_index_with_chunk()

            i, t = chunk.get_by_index(id)

            inputs.append(i)
            targets.append(t)

        return torch.stack(inputs, dim=1), torch.stack(targets, dim=1)
