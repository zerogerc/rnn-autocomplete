import os
import re

import torch

from zerogercrnn.lib.utils.split import split_data

DEFAULT_ENCODING = 'ISO-8859-1'


def get_alphabet(directory, pattern='((train)|(validation)|(test)).txt'):
    """Scans files in the directory that match regular expression 
    and creates alphabet of all characters encountered during scan.
    """
    all_letters = set()
    all_letters.add('\n')
    all_letters.add('\t')

    reg = re.compile(pattern)
    for file in os.listdir(directory):
        if reg.match(file):
            # Add letters to the dictionary
            with open(os.path.join(directory, file), 'r', encoding=DEFAULT_ENCODING) as f:
                for line in f:
                    for c in line:
                        all_letters.add(c)

    return ''.join(sorted(all_letters))


def create_char_to_idx_and_backward(alphabet):
    """Create map from letter to position in the alphabet and array of letters in the order of alphabet."""
    char2idx = {}
    idx2char = []
    for c in alphabet:
        char2idx[c] = len(idx2char)
        idx2char.append(c)
    return char2idx, idx2char


def tokenize(path, alphabet):
    """Tokenizes a text file. Replaces all letters with their indexes in alphabet."""

    assert os.path.exists(path)
    char2idx, idx2char = create_char_to_idx_and_backward(alphabet=alphabet)

    # Calculate number of tokens to create tensor with the same length.
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
                ids[token] = char2idx[c]
                token += 1

            ids[token] = char2idx['\n']
            token += 1

    return ids


def tokenize_single_without_alphabet(path):
    """Creates alphabet from given file and tokenize it. This function is dangerous use it only for test purposes."""
    all_letters = set()
    all_letters.add('\n')
    all_letters.add('\t')

    with open(path, 'r', encoding=DEFAULT_ENCODING) as f:
        for line in f:
            for c in line:
                all_letters.add(c)

    alphabet = ''.join(sorted(all_letters))
    return tokenize(path, alphabet=alphabet), alphabet


class Corpus:
    """Class that stores tokenized version of train/valid/test datasets using character-level dictionary."""

    def __init__(self, train, valid, test, alphabet):
        self.alphabet = alphabet
        self.train = train
        self.valid = valid
        self.test = test

    @staticmethod
    def create_from_data_dir(path, alphabet):
        """Creates instance of Corpus. Supposes that train/validation/test datasets is stored in the path 
        with names train.txt/validation.txt/test.txt.
        
        :param path: full path to the directory with files
        :param alphabet: string of all possible symbols in text
        """

        train_file = os.path.join(path, 'train.txt')
        validation_file = os.path.join(path, 'validation.txt')
        test_file = os.path.join(path, 'test.txt')

        train = tokenize(train_file, alphabet=alphabet)
        valid = tokenize(validation_file, alphabet=alphabet)
        test = tokenize(test_file, alphabet=alphabet)

        return Corpus(train=train, valid=valid, test=test, alphabet=alphabet)

    @staticmethod
    def create_from_single_file(path, *, validation_percentage=0.1, test_percentage=0.1, alphabet=None):
        """Creates instance of Corpus using a single file. **Use it for testing purposes only.**
        File will be split on train/validation/test using
        specified percentages.
        
        :param path: full path to the files with data
        :param validation_percentage: percentage of validation data
        :param test_percentage: percentage of test data
        :param alphabet: all possible characters, if None alphabet will be created from file
        """
        if alphabet is None:
            single, alphabet = tokenize_single_without_alphabet(path)
        else:
            single = tokenize(path, alphabet=alphabet)

        input_tensor = single.unsqueeze(1)
        train, valid, test = split_data(
            data_tensor=input_tensor,
            validation_percentage=validation_percentage,
            test_percentage=test_percentage,
            shuffle=False
        )

        return Corpus(train=train, valid=valid, test=test, alphabet=alphabet)
