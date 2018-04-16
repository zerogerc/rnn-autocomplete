import json
import torch

from zerogercrnn.lib.data.programs_batch import DataChunk, BatchedDataGenerator, split_train_validation
from zerogercrnn.lib.data.general import DataReader, DataGenerator
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils.time import tqdm_lim

ENCODING = 'ISO-8859-1'


class ASTOneHotChunk(DataChunk):
    def __init__(self, data_one_hot, cuda, number_of_seq):
        self.data_one_hot = data_one_hot
        self.cuda = cuda
        self.number_of_seq = number_of_seq

        self.seq_len = None

    def prepare_data(self, seq_len):
        self.seq_len = seq_len

        ln = self.size() - self.size() % seq_len
        if ln == 0:
            raise Exception('Chunk is too small. Consider filtering it out')

        self.data_one_hot = self.data_one_hot.narrow(dimension=0, start=0, length=ln)

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

    def __init__(self, non_terminals_one_hot, terminals_one_hot, cuda, number_of_seq):
        self.non_terminals_chunk = ASTOneHotChunk(
            data_one_hot=non_terminals_one_hot,
            cuda=cuda,
            number_of_seq=number_of_seq
        )

        self.terminals_chunk = ASTOneHotChunk(
            data_one_hot=terminals_one_hot,
            cuda=cuda,
            number_of_seq=number_of_seq
        )

        assert self.non_terminals_chunk.size() == self.terminals_chunk.size()

    def prepare_data(self, seq_len):
        self.non_terminals_chunk.prepare_data(seq_len)
        self.terminals_chunk.prepare_data(seq_len)
        assert self.non_terminals_chunk.size() == self.terminals_chunk.size()

    def get_by_index(self, index):
        non_terminals_input, non_terminals_target = self.non_terminals_chunk.get_by_index(index)
        terminals_input, terminals_target = self.terminals_chunk.get_by_index(index)

        return (non_terminals_input, terminals_input), (non_terminals_target, terminals_target)

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
            self.eval_data = self._read_programs(file_eval, total=50000, limit=limit)

    def _read_programs(self, file, total, limit):
        chunks = []
        with open(file, mode='r', encoding=ENCODING) as f:
            for line in tqdm_lim(f, total=total, lim=limit):
                nodes = json.loads(line)

                non_terminals_one_hot = torch.LongTensor(len(nodes))
                terminals_one_hot = torch.LongTensor(len(nodes))

                it = 0
                for node in nodes:
                    non_terminals_one_hot[it] = int(node['N'])
                    terminals_one_hot[it] = int(node['T'])
                    it += 1

                chunks.append(ASTDataChunk(
                    non_terminals_one_hot=non_terminals_one_hot,
                    terminals_one_hot=terminals_one_hot,
                    cuda=self.cuda,
                    number_of_seq=self.number_of_seq
                ))

        return list(filter(lambda d: d.size() >= self.seq_len, chunks))


class ASTDataGenerator(BatchedDataGenerator):

    def __init__(self, data_reader, seq_len, batch_size, cuda):
        super().__init__(data_reader, seq_len, batch_size, cuda)

    # TODO: optimize
    def _retrieve_batch(self, key, buckets):
        nt_inputs = []
        t_inputs = []
        nt_targets = []
        t_targets = []

        for b in buckets:
            id, chunk = b.get_next_index_with_chunk()
            (nt_input, t_input), (nt_target, t_target) = chunk.get_by_index(id)

            nt_inputs.append(nt_input)
            t_inputs.append(t_input)
            nt_targets.append(nt_target)
            t_targets.append(t_target)

        nt_inputs = torch.stack(nt_inputs, dim=1)
        t_inputs = torch.stack(t_inputs, dim=1)
        nt_targets = torch.stack(nt_targets, dim=1)
        t_targets = torch.stack(t_targets, dim=1)

        return (nt_inputs, t_inputs), (nt_targets, t_targets)


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
