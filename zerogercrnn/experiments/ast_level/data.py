import json

import torch
import numpy as np

from zerogercrnn.lib.calculation import pad_tensor
from zerogercrnn.lib.data import DataChunk, BatchedDataGenerator, split_train_validation, DataReader
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.log import tqdm_lim
from zerogercrnn.lib.utils import setup_tensor

ENCODING = 'ISO-8859-1'


class ASTInput:
    def __init__(self, non_terminals, terminals, nodes_depth=None):
        self.non_terminals = non_terminals
        self.terminals = terminals
        self.current_non_terminals = None
        self.nodes_depth = nodes_depth

    @staticmethod
    def setup(input_data, cuda):
        """Returns new ASTInput with tensors located on the needed devices."""
        return ASTInput(
            non_terminals=setup_tensor(input_data.non_terminals, cuda),
            terminals=setup_tensor(input_data.terminals, cuda),
            nodes_depth=input_data.nodes_depth  # no gradients should be computed
        )

    @staticmethod
    def combine(inputs, dim):
        """Combine several inputs to batch adding one extra dim at position dim."""
        non_terminals_combined = torch.stack([i.non_terminals for i in inputs], dim=dim)
        terminals_combined = torch.stack([i.terminals for i in inputs], dim=dim)
        depths_combined = torch.stack([i.nodes_depth for i in inputs], dim=dim)

        return ASTInput(non_terminals_combined, terminals_combined, depths_combined)


class ASTTarget:
    def __init__(self, non_terminals, terminals):
        self.non_terminals = non_terminals
        self.terminals = terminals

    @staticmethod
    def setup(target_data, cuda):
        """Returns new ASTTarget with tensors located on the needed devices."""
        return ASTTarget(
            non_terminals=setup_tensor(target_data.non_terminals, cuda),
            terminals=setup_tensor(target_data.terminals, cuda)
        )

    @staticmethod
    def combine(inputs, dim):
        """Combine several targets to batch adding one extra dim at position dim."""
        non_terminals_combined = torch.stack([i.non_terminals for i in inputs], dim=dim)
        terminals_combined = torch.stack([i.terminals for i in inputs], dim=dim)

        return ASTTarget(non_terminals_combined, terminals_combined)


class TensorData:
    """Class that holds tensor. Can pad tensor with last element and safely retrieve data by index."""

    def __init__(self, data, cuda):
        self.data = data
        self.cuda = cuda

    def prepare_data(self, seq_len):
        self.data = pad_tensor(self.data, seq_len=seq_len)

    def get_by_index(self, index, seq_len):
        if index + seq_len > self.size():
            raise Exception('Not enough data! index: {}, seq_len: {}'.format(index, seq_len))

        return self.data.narrow(dim=0, start=index, length=seq_len)

    def size(self):
        return self.data.size()[0]


class ASTOneHotChunk(DataChunk):
    def __init__(self, data_one_hot, cuda):
        self.data_one_hot = data_one_hot
        self.cuda = cuda

        self.seq_len = None

    def prepare_data(self, seq_len):
        self.seq_len = seq_len
        self.data_one_hot = pad_tensor(self.data_one_hot, seq_len=seq_len)

    def get_by_index(self, index):
        if self.seq_len is None:
            raise Exception('You should call prepare_data with specified seq_len first')
        if index + self.seq_len > self.size():
            raise Exception('Not enough data in chunk')

        input_tensor = self.data_one_hot.narrow(dim=0, start=index, length=self.seq_len - 1)
        target_tensor = self.data_one_hot.narrow(dim=0, start=index + 1, length=self.seq_len - 1)

        return input_tensor, target_tensor

    def size(self):
        return self.data_one_hot.size()[0]


class ASTDataChunk(DataChunk):

    def __init__(self, non_terminals_one_hot, terminals_one_hot, nodes_depth, cuda):
        self.seq_len = None
        self.non_terminals_chunk = ASTOneHotChunk(
            data_one_hot=non_terminals_one_hot,
            cuda=cuda
        )

        self.terminals_chunk = ASTOneHotChunk(
            data_one_hot=terminals_one_hot,
            cuda=cuda
        )

        self.nodes_depth_data = TensorData(data=nodes_depth, cuda=cuda)

        assert self.non_terminals_chunk.size() == self.terminals_chunk.size()

    def prepare_data(self, seq_len):
        self.seq_len = seq_len
        self.non_terminals_chunk.prepare_data(seq_len)
        self.terminals_chunk.prepare_data(seq_len)
        self.nodes_depth_data.prepare_data(seq_len)
        assert self.non_terminals_chunk.size() == self.terminals_chunk.size()
        assert self.terminals_chunk.size() == self.nodes_depth_data.size()

    def get_by_index(self, index):
        if self.seq_len is None:
            raise Exception('You should call prepare_data first.')
        non_terminals_input, non_terminals_target = self.non_terminals_chunk.get_by_index(index)
        terminals_input, terminals_target = self.terminals_chunk.get_by_index(index)
        nodes_depth_input = self.nodes_depth_data.get_by_index(index, seq_len=self.seq_len - 1)

        return ASTInput(non_terminals_input, terminals_input, nodes_depth_input), \
               ASTTarget(non_terminals_target, terminals_target)

    def size(self):
        return self.non_terminals_chunk.size()


class ASTDataReader(DataReader):

    def __init__(self, file_train, file_eval, cuda, seq_len, number_of_seq=20, limit=None):
        super().__init__()
        self.cuda = cuda
        self.seq_len = seq_len
        self.number_of_seq = number_of_seq

        if file_train is not None:
            self.train_data, self.validation_data = split_train_validation(
                self._read_programs(file_train, total=100000, limit=limit),
                split_coefficient=0.8
            )

        if file_eval is not None:
            self.eval_data, self.eval_tails = self._read_programs(file_eval, total=50000, limit=limit, count_tails=True)

    def _read_programs(self, file, total, limit, count_tails=False):
        chunks = []
        tails = 0

        with open(file, mode='r', encoding=ENCODING) as f:
            for line in tqdm_lim(f, total=total, lim=limit):
                nodes = json.loads(line)

                non_terminals_one_hot = np.empty(len(nodes), dtype=int)
                terminals_one_hot = np.empty(len(nodes), dtype=int)
                nodes_depth = np.empty(len(nodes), dtype=int)

                it = 0
                for node in nodes:
                    non_terminals_one_hot[it] = int(node['N'])
                    terminals_one_hot[it] = int(node['T'])
                    nodes_depth[it] = int(node['d'])
                    it += 1

                tails += len(nodes) % self.seq_len  # this is the size of appended tails <EOF, EMP>
                chunks.append(ASTDataChunk(
                    non_terminals_one_hot=torch.tensor(non_terminals_one_hot, dtype=torch.long),
                    terminals_one_hot=torch.tensor(terminals_one_hot, dtype=torch.long),
                    nodes_depth=torch.tensor(nodes_depth, dtype=torch.long),
                    cuda=self.cuda
                ))

        if count_tails:
            return chunks, tails
        else:
            return chunks


class ASTDataGenerator(BatchedDataGenerator):

    def __init__(self, data_reader, seq_len, batch_size, cuda):
        super().__init__(data_reader, seq_len, batch_size, cuda)

    def _retrieve_batch(self, key, buckets):
        inputs = []
        targets = []

        for b in buckets:
            id, chunk = b.get_next_index_with_chunk()
            cur_input, cur_target = chunk.get_by_index(id)

            inputs.append(cur_input)
            targets.append(cur_target)
        return ASTInput.combine(inputs, dim=1), ASTTarget.combine(targets, dim=1)


if __name__ == '__main__':
    file_train = 'data/ast/file_train.json'
    vectors_file = 'data/ast/vectors.txt'

    embeddings = Embeddings(embeddings_size=50, vector_file='data/ast/vectors.txt', squeeze=True)
    data_reader = ASTDataReader(
        file_train=file_train,
        file_eval=None,
        cuda=False,
        seq_len=10,
        number_of_seq=20,
        limit=200
    )
    data_generator = ASTDataGenerator(
        data_reader=data_reader,
        seq_len=10,
        batch_size=10,
        cuda=False
    )

    its = 0
    for iter_data in data_generator.get_train_generator():
        print(iter_data[0][0])
        print(iter_data[0][1])
        print(iter_data[1][0])
        print(iter_data[1][1])
        its += 1

    print(its)
