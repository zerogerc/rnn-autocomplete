import json
import torch

from zerogercrnn.lib.data.general import DataReader
from zerogercrnn.lib.data.programs_batch import DataChunk, BatchedDataGenerator, split_train_validation
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.log import tqdm_lim
from zerogercrnn.lib.utils import wrap_cuda_no_grad_variable

ENCODING = 'ISO-8859-1'


def pad_tensor(tensor, seq_len, cuda):
    """Pad tensor with last element."""
    assert len(tensor.size()) == 1

    tail = torch.LongTensor([tensor[-1]]).expand(seq_len - tensor.size()[0] % seq_len)
    tensor = torch.cat((tensor, tail))

    assert tensor.size()[0] % seq_len == 0
    if cuda:
        tensor = tensor.cuda()

    return tensor


class ASTInput:
    def __init__(self, non_terminals, terminals, nodes_depth=None):
        self.non_terminals = non_terminals
        self.terminals = terminals
        self.nodes_depth = nodes_depth

    @staticmethod
    def wrap_cuda_no_grad(input_data, cuda, no_grad):
        """Returns new ASTInput that fields are variables."""
        return ASTInput(
            non_terminals=wrap_cuda_no_grad_variable(input_data.non_terminals, cuda, no_grad),
            terminals=wrap_cuda_no_grad_variable(input_data.terminals, cuda, no_grad),
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
    def wrap_cuda_no_grad(target_data, cuda, no_grad):
        """Returns new ASTTarget that fields are variables."""
        return ASTTarget(
            non_terminals=wrap_cuda_no_grad_variable(target_data.non_terminals, cuda, no_grad),
            terminals=wrap_cuda_no_grad_variable(target_data.terminals, cuda, no_grad)
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
        self.data = pad_tensor(self.data, seq_len=seq_len, cuda=self.cuda)

    def get_by_index(self, index, seq_len):
        if index + seq_len > self.size():
            raise Exception('Not enough data! index: {}, seq_len: {}'.format(index, seq_len))

        return self.data.narrow(dimension=0, start=index, length=seq_len)

    def size(self):
        return self.data.size()[0]


class ASTOneHotChunk(DataChunk):
    def __init__(self, data_one_hot, cuda):
        self.data_one_hot = data_one_hot
        self.cuda = cuda

        self.seq_len = None

    def prepare_data(self, seq_len):
        assert len(self.data_one_hot.size()) == 1
        self.seq_len = seq_len

        tail = torch.LongTensor([self.data_one_hot[-1]]).expand(self.seq_len - self.size() % self.seq_len)
        self.data_one_hot = torch.cat((self.data_one_hot, tail))
        assert self.size() % seq_len == 0
        if self.cuda:
            self.data_one_hot = self.data_one_hot.cuda()

    def get_by_index(self, index):
        if self.seq_len is None:
            raise Exception('You should call prepare_data with specified seq_len first')
        if index + self.seq_len > self.size():
            raise Exception('Not enough data in chunk')

        input_tensor = self.data_one_hot.narrow(dimension=0, start=index, length=self.seq_len - 1)
        target_tensor = self.data_one_hot.narrow(dimension=0, start=index + 1, length=self.seq_len - 1)

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
        nodes_depth_input = self.nodes_depth_data.get_by_index(index, seq_len=self.seq_len)

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

                non_terminals_one_hot = torch.LongTensor(len(nodes))
                terminals_one_hot = torch.LongTensor(len(nodes))
                nodes_depth = torch.LongTensor(len(nodes))

                it = 0
                for node in nodes:
                    non_terminals_one_hot[it] = int(node['N'])
                    terminals_one_hot[it] = int(node['T'])
                    nodes_depth[it] = int(node['d'])
                    it += 1

                tails += len(nodes) % self.seq_len  # this is the size of appended tails <EOF, EMP>
                chunks.append(ASTDataChunk(
                    non_terminals_one_hot=non_terminals_one_hot,
                    terminals_one_hot=terminals_one_hot,
                    nodes_depth=nodes_depth,
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
