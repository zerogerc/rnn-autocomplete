import os
import torch

from global_constants import DEFAULT_ENCODING


class Corpus:
    """Tokenizes text from train/valid/test datasets using character-level dictionary."""

    def __init__(self, path, *, single_file=None, letters=None):
        self.all_letters = set()
        self.char2idx = {}
        self.idx2char = []

        if single_file is None:
            train_file = os.path.join(path, 'train.txt')
            validation_file = os.path.join(path, 'validation.txt')
            test_file = os.path.join(path, 'test.txt')

            if letters is None:
                self.__create_dicts__(train_file, validation_file, test_file)
            else:
                self.__char_to_idx__(letters)

            self.train = self.tokenize(train_file)
            self.valid = self.tokenize(validation_file)
            self.test = self.tokenize(test_file)
        else:
            file = os.path.join(path, single_file)
            if letters is None:
                self.__create_dicts__(file)
            else:
                self.__char_to_idx__(letters)
            self.single = self.tokenize(file)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add letters to the dictionary
        with open(path, 'r', encoding=DEFAULT_ENCODING) as f:
            tokens = 0
            for line in f:
                tokens += len(line) + 1  # +1 for \n

        # Tokenize file content
        with open(path, 'r', encoding=DEFAULT_ENCODING) as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                for c in line:
                    ids[token] = self.char2idx[c]
                    token += 1

                ids[token] = self.char2idx['\n']
                token += 1

        return ids

    def __create_dicts__(self, *paths):
        for path in paths:
            assert os.path.exists(path)
            # Add letters to the dictionary
            with open(path, 'r', encoding=DEFAULT_ENCODING) as f:
                tokens = 0
                for line in f:
                    tokens += len(line) + 1  # +1 for \n
                    for c in line:
                        self.all_letters.add(c)

        self.all_letters.add('\n')

        letters = ''.join(sorted(self.all_letters))
        self.__char_to_idx__(letters)

    def __char_to_idx__(self, letters):
        self.all_letters = set(letters)
        for c in letters:
            self.char2idx[c] = len(self.idx2char)
            self.idx2char.append(c)


if __name__ == '__main__':
    corpus = Corpus(path='/Users/zerogerc/Yandex.Disk.localized/shared_files/University/rnn-autocomplete/data')
    print(corpus.train.size())
