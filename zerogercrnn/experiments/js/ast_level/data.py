import json
import os

import numpy as np
import torch
from tqdm import tqdm

from zerogercrnn.experiments.js.ast_level.raw_data import ENCODING
from zerogercrnn.lib.data.general import DataGenerator

"""
File that contains utilities that transform sequence of one-hot (Terminal, Non-Terminal) 
to the tensors that could be fed into models.
"""

DIR_DATASET = '/Users/zerogerc/Documents/datasets/js_dataset.tar/processed'

FILE_TRAINING = os.path.join(DIR_DATASET, 'programs_training_one_hot.json')
FILE_EVAL = os.path.join(DIR_DATASET, 'programs_eval_one_hot.json')

FILE_TERMINALS = os.path.join(DIR_DATASET, 'terminal_tokens.txt')
FILE_NON_TERMINALS = os.path.join(DIR_DATASET, 'non_terminal_tokens.txt')


class SourceFile:
    """Encoded source file of JS AST. Has two fields N and T that are the same length.
    N is a sequence of non-terminals and T is a sequence of corresponding terminals.

    N: torch.LongTensor
    T: torch.LongTensor
    """

    def __init__(self, N: torch.LongTensor, T: torch.LongTensor):
        assert N.size() == T.size()
        self.N = N
        self.T = T

    def size(self):
        return self.N.size()


class ASTDataGenerator(DataGenerator):
    """Provides batched data for training and evaluation of model."""

    def __init__(self, data_reader, seq_len, batch_size):
        super(ASTDataGenerator, self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.data_train = self._prepare_data_(data_reader.data_train)
        self.data_eval = self._prepare_data_(data_reader.data_eval)

        self.buckets = []
        for i in range(self.batch_size):
            self.buckets.append(DataBucket(self.seq_len))

    # override
    def get_train_generator(self):
        return self._get_batched_epoch_(dataset=self.data_train)

    # override
    def get_validation_generator(self):
        return self._get_batched_epoch_(dataset=self.data_eval, limit=1000)

    def _get_random_batch_(self, dataset):
        """Returns batch of first sequences of random files.
            TODO: enhance. now it's r
        """
        for i in range(self.batch_size):
            self.buckets[i].clear()
            self.buckets[i].add_first_seq_of_source_file(dataset[ASTDataGenerator._get_random_index(len(dataset))])

        yield self._retrieve_batch_()

    def _get_batched_epoch_(self, dataset, limit=None):
        """Returns generator over batched data of all files in the dataset."""
        indexes = ASTDataGenerator._get_shuffled_indexes_(len(dataset))
        current = 0
        while True:
            cont = True  # indicates if epoch is finished
            for bucket in self.buckets:  # add SourceFiles from query to the empty buckets.
                if bucket.is_empty():
                    if current == len(indexes):
                        cont = False
                        break

                    bucket.add_source_file(source=dataset[indexes[current]])  # add source file to fill the bucket
                    current += 1

            if cont:
                yield self._retrieve_batch_()
            else:
                break

            if current == limit:
                break

    def _retrieve_batch_(self):
        """Returns pair of two tensors for input and target: (N, T) with sizes [seq_len, batch_size].
            All buckets should not be empty here.

            TODO: optimize
        """
        N = []
        T = []

        for bucket in self.buckets:
            n, t = bucket.get_next_seq()
            N.append(n.unsqueeze(1))
            T.append(t.unsqueeze(1))

        N_cat = torch.cat(seq=N, dim=1)
        T_cat = torch.cat(seq=T, dim=1)

        t_input = (N_cat[0:self.seq_len, :], T_cat[0:self.seq_len, :])
        t_target = (N_cat[1:self.seq_len + 1, :], T_cat[1:self.seq_len + 1, :])
        return t_input, t_target

    def _prepare_data_(self, data):
        """Add (EOF, EMPTY) tokens to the end of file to match with self.seq_len. """
        for i in tqdm(range(len(data))):
            source_file_n = data[i].N
            source_file_t = data[i].T
            assert source_file_n.size() == source_file_t.size()

            tail_size = (self.seq_len - source_file_n.size()[0] % self.seq_len) + 1  # + 1 for target
            tail_n = torch.LongTensor([source_file_n[-1]]).expand(tail_size)
            tail_t = torch.LongTensor([source_file_t[-1]]).expand(tail_size)

            if reader.cuda:
                tail_n = tail_n.cuda()
                tail_t = tail_t.cuda()

            assert tail_n.size() == tail_t.size()

            data[i].N = torch.cat((data[i].N, tail_n))
            data[i].T = torch.cat((data[i].T, tail_t))

        return data

    @staticmethod
    def _get_shuffled_indexes_(length):
        temp = np.arange(length)
        np.random.shuffle(temp)
        return temp

    @staticmethod
    def _get_random_index(length):
        return np.random.randint(length)


class DataBucket:
    """Bucket with SourceFiles. You could put SourceFiles here and extract tensors of (N, T) with specified seq_len."""

    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.source = None
        self.index = 0

    def is_empty(self):
        """Indicates whether this bucket contains at least one more sequence."""
        return (self.source is None) or ((self.source.size()[0] - 1) == self.index)

    def clear(self):
        """Remove attached SourceFile from this bucket."""
        self.source = None
        self.index = 0

    def get_next_seq(self):
        """Return two values of size (seq_len + 1) to be able to get input and target:
            N - is the sequence of non-terminals,
            T - is the sequence of corresponding terminals.
        """
        assert not self.is_empty()
        self.index += self.seq_len
        start = self.index - self.seq_len
        return self.source.N.narrow(0, start, self.seq_len + 1), self.source.T.narrow(0, start, self.seq_len + 1)

    def add_source_file(self, source: SourceFile):
        """Adds the whole source file to the bucket."""
        assert (source.N.size()[0] - 1) % self.seq_len == 0
        assert (source.T.size()[0] - 1) % self.seq_len == 0
        self.source = source
        self.index = 0

    def add_first_seq_of_source_file(self, source: SourceFile):
        """Adds the first segment of the source file to the bucket."""
        assert (source.N.size()[0] - 1) % self.seq_len == 0
        assert (source.T.size()[0] - 1) % self.seq_len == 0
        self.source = SourceFile(
            N=source.N[0:self.seq_len + 1],
            T=source.T[0:self.seq_len + 1]
        )
        self.index = 0


class MockDataReader:
    def __init__(self, cuda=True):
        self.cuda = cuda and torch.cuda.is_available()

        N = torch.LongTensor([1] * 1000)
        T = torch.LongTensor(np.arange(1000))

        if self.cuda:
            N = N.cuda()
            T = T.cuda()

        self.data_train = [SourceFile(N, T) for i in range(100)]
        self.data_eval = [SourceFile(N, T) for i in range(100)]


class DataReader:
    """Reads the data from one-hot encoded files and stores them as array of SourceFiles."""

    def __init__(self, file_training, file_eval, encoding=ENCODING, limit_train=None, limit_eval=None, cuda=True):
        self.cuda = cuda and torch.cuda.is_available()

        data_train_limit = 100000 if (limit_train is None) else limit_train
        data_eval_limit = 50000 if (limit_eval is None) else limit_eval

        self.data_train = self.parse_programs(
            file_training,
            encoding,
            total=data_train_limit,
            label='Train',
            cuda=self.cuda
        )
        self.data_eval = self.parse_programs(
            file_eval,
            encoding,
            total=data_eval_limit,
            label='Eval',
            cuda=self.cuda
        )

    @staticmethod
    def parse_programs(file, encoding, total, label, cuda):
        print('Parsing {}'.format(label))

        programs = []
        with open(file, mode='r', encoding=encoding) as f:
            cur = 0
            for l in tqdm(f, total=total):
                raw_json = json.loads(l)

                N = []
                T = []
                for node in raw_json:
                    assert ('N' in node) and ('T' in node)
                    N.append(node['N'])
                    T.append(node['T'])

                # TODO: check if it's faster to allocated tensors on GPU right away
                tensorN = torch.LongTensor(N)
                tensorT = torch.LongTensor(T)

                if cuda:
                    tensorN = tensorN.cuda()
                    tensorT = tensorT.cuda()

                programs.append(
                    SourceFile(
                        N=torch.LongTensor(N),
                        T=torch.LongTensor(T)
                    )
                )

                cur += 1
                if cur == total:
                    break

        return programs


if __name__ == '__main__':
    reader = DataReader(
        file_training=FILE_TRAINING,
        file_eval=FILE_EVAL,
        encoding=ENCODING
    )
