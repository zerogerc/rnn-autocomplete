import json
import torch

from zerogercrnn.lib.data.programs_batch import DataChunk, BatchedDataGenerator, split_train_validation
from zerogercrnn.lib.data.general import DataReader, DataGenerator
from zerogercrnn.lib.embedding import Embeddings

ENCODING = 'ISO-8859-1'


class ASTTerminalsEmbeddedChunk(DataChunk):
    def __init__(self, terminals_one_hot, embeddings: Embeddings, cuda, number_of_seq):
        self.terminals_one_hot = terminals_one_hot
        self.embeddings = embeddings
        self.cuda = cuda
        self.number_of_seq = number_of_seq

        self.embeddings_cache = None
        self.seq_len = None

    def prepare_data(self, seq_len):
        self.seq_len = seq_len

        ln = min(self.size() - self.size() % seq_len, self.number_of_seq * self.seq_len)
        if ln == 0:
            raise Exception('Chunk is too small. Consider filtering it out')

        self.terminals_one_hot = self.terminals_one_hot.narrow(dimension=0, start=0, length=ln)

        if self.cuda:
            self.terminals_one_hot = self.terminals_one_hot.cuda()

    def init_cache_if_needed(self, cur_index):
        if cur_index != 0:
            return

        self.embeddings_cache = self.embeddings.index_select(self.terminals_one_hot)
        if self.cuda:
            self.embeddings_cache = self.embeddings_cache.cuda()

    def drop_cache_if_needed(self, cur_index):
        if cur_index + 2 * self.seq_len > self.size():
            self.embeddings_cache = None

    def get_by_index(self, index):
        if self.seq_len is None:
            raise Exception('You should call prepare_data with specified seq_len first')
        if index + self.seq_len > self.size():
            raise Exception('Not enough data in chunk')
        self.init_cache_if_needed(index)

        input_tensor_emb = self.embeddings_cache.narrow(dimension=0, start=index, length=self.seq_len - 1)
        target_tensor = self.terminals_one_hot.narrow(dimension=0, start=index + 1, length=self.seq_len - 1)

        self.drop_cache_if_needed(index)
        return input_tensor_emb, target_tensor

    def size(self):
        return self.terminals_one_hot.size()[0]


class ASTNonTerminalsOneHotChunk(DataChunk):
    def __init__(self, non_terminals_one_hot, cuda, number_of_seq):
        self.non_terminals_one_hot = non_terminals_one_hot
        self.cuda = cuda
        self.number_of_seq = number_of_seq

        self.seq_len = None

    def prepare_data(self, seq_len):
        self.seq_len = seq_len

        ln = min(self.size() - self.size() % seq_len, self.number_of_seq * self.seq_len)
        if ln == 0:
            raise Exception('Chunk is too small. Consider filtering it out')

        self.non_terminals_one_hot = self.non_terminals_one_hot.narrow(dimension=0, start=0, length=ln)

        if self.cuda:
            self.non_terminals_one_hot = self.non_terminals_one_hot.cuda()

    def get_by_index(self, index):
        if self.seq_len is None:
            raise Exception('You should call prepare_data with specified seq_len first')
        if index + self.seq_len > self.size():
            raise Exception('Not enough data in chunk')

        input_tensor_emb = self.non_terminals_one_hot.narrow(dimension=0, start=index, length=self.seq_len - 1)
        target_tensor = self.non_terminals_one_hot.narrow(dimension=0, start=index + 1, length=self.seq_len - 1)

        return input_tensor_emb, target_tensor

    def size(self):
        return self.non_terminals_one_hot.size()[0]


class ASTDataChunk(DataChunk):

    def __init__(self, non_terminals_one_hot, terminals_one_hot, embeddings: Embeddings, cuda, number_of_seq):
        self.non_terminals_chunk = ASTNonTerminalsOneHotChunk(
            non_terminals_one_hot=non_terminals_one_hot,
            cuda=cuda,
            number_of_seq=number_of_seq
        )

        self.terminals_chunk = ASTTerminalsEmbeddedChunk(
            terminals_one_hot=terminals_one_hot,
            embeddings=embeddings,
            cuda=cuda,
            number_of_seq=number_of_seq
        )

        assert self.non_terminals_chunk.size() == self.terminals_chunk.size()

    def prepare_data(self, seq_len):
        self.non_terminals_chunk.prepare_data(seq_len)
        self.terminals_chunk.prepare_data(seq_len)
        assert self.non_terminals_chunk.size() == self.terminals_chunk.size()

    def init_cache_if_needed(self, cur_index):
        self.terminals_chunk.init_cache_if_needed(cur_index=cur_index)

    def drop_cache_if_needed(self, cur_index):
        self.terminals_chunk.drop_cache_if_needed(cur_index=cur_index)

    def get_by_index(self, index):
        non_terminals_input, non_terminals_target = self.non_terminals_chunk.get_by_index(index)
        terminals_input, terminals_target = self.terminals_chunk.get_by_index(index)

        return (non_terminals_input, terminals_input), (non_terminals_target, terminals_target)

    def size(self):
        return self.non_terminals_chunk.size()


class ASTLevelDataReader(DataReader):

    def __init__(self, file_train, file_eval, embeddings: Embeddings, cuda, number_of_seq=20, limit=None):
        super().__init__()
        self.embeddings = embeddings
        self.cuda = cuda
        self.number_of_seq = number_of_seq

        if file_train is not None:
            self.train_data, self.validation_data = split_train_validation(
                self._read_programs(file_train, limit=limit),
                split_coefficient=0.8
            )

        if file_eval is not None:
            self.eval_data = self._read_programs(file_eval, limit=limit)

    def _read_programs(self, file, limit):
        chunks = []
        with open(file, mode='r', encoding=ENCODING) as f:
            for line in f:
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
                    embeddings=self.embeddings,
                    cuda=self.cuda,
                    number_of_seq=self.number_of_seq
                ))


if __name__ == '__main__':
    file_train = 'data/ast/file_train.json'
    vectors_file = 'data/ast/vectors.txt'

