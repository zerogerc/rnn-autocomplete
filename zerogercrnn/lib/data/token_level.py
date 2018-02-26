"""
Token-level parsing of input.
"""
import os
import torch

from zerogercrnn.experiments.linux.constants import HOME_DIR
from zerogercrnn.lib.utils.split import split_data

DEFAULT_ENCODING = 'ISO-8859-1'

UNK_TKN = '_UNK_'
NEWLINE_TKN = '\n'


class Corpus:
    def __init__(self, train, valid, test, tokens):
        self.tokens = tokens
        self.train = train
        self.valid = valid
        self.test = test

    @staticmethod
    def create_from_data_dir(path, tokens_path):
        """Creates instance of Corpus. Supposes that train/validation/test datasets is stored in the path 
        with names train.txt/validation.txt/test.txt.

        :param path: full path to the directory with files
        :param alphabet: string of all possible symbols in text
        """
        validation_file = os.path.join(path, 'validation.txt')
        test_file = os.path.join(path, 'test.txt')
        train_file = os.path.join(path, 'train.txt')

        valid, tokens = tokenize_file(path=validation_file, tokens_path=tokens_path)
        test, tokens = tokenize_file(path=test_file, tokens_path=tokens_path)
        train, tokens = tokenize_file(path=train_file, tokens_path=tokens_path)

        return Corpus(train=train, valid=valid, test=test, tokens=tokens)

    @staticmethod
    def create_from_single_file(path, tokens_path, *, validation_percentage=0.1, test_percentage=0.1):
        single, tokens = tokenize_file(path=path, tokens_path=tokens_path)

        input_tensor = single.unsqueeze(1)
        train, valid, test = split_data(
            data_tensor=input_tensor,
            validation_percentage=validation_percentage,
            test_percentage=test_percentage,
            shuffle=False
        )

        return Corpus(train=train, valid=valid, test=test, tokens=tokens)


def tokenize_file(path, tokens_path):
    tokens = _read_tokens_(path=tokens_path)

    tokens.append(NEWLINE_TKN)
    tokens.append(UNK_TKN)

    token2idx, idx2token = _create_to_from_(tokens)

    # number_of_tokens = _get_number_of_tokens_(path)

    return _tokenize_(path, number_of_tokens=0, token2idx=token2idx, idx2token=idx2token), tokens


def _tokenize_(path, number_of_tokens, token2idx, idx2token):
    with open(path, 'r', encoding=DEFAULT_ENCODING) as f:
        ids = []
        for line in f:
            i = 0
            while i < len(line):
                tkn, i = _next_token_(line, i)

                if tkn in token2idx:
                    ids.append(token2idx[tkn])
                else:
                    ids.append(token2idx[UNK_TKN])

    return torch.LongTensor(ids)


def _create_to_from_(tokens):
    token2idx = {}
    for i in range(len(tokens)):
        tkn = tokens[i]
        token2idx[tkn] = i
    return token2idx, tokens


def _read_tokens_(path):
    tokens = []
    with open(file=path, mode='r', encoding=DEFAULT_ENCODING) as f:
        for line in f:
            if line[-1] == '\n':
                tokens.append(line[:-1])
            else:
                tokens.append(line)
    return tokens


def get_and_write_tokens(path, limit_occur, output_file=None):
    tokens, token_occ = _get_tokens_from_file_(path=path)

    out_tokens = set()
    for tkn in tokens:
        if token_occ[tkn] >= limit_occur:
            out_tokens.add(tkn)

    if output_file is not None:
        with open(file=os.path.join(os.getcwd(), 'tokens.txt'), mode='w', encoding=DEFAULT_ENCODING) as f:
            for token in out_tokens:
                if token != '\n':
                    f.write('{}\n'.format(token))


def _get_tokens_from_file_(path):
    assert os.path.exists(path)

    tokens = set()
    token_occ = {}

    # Calculate number of tokens to create tensor with the same length.
    with open(path, 'r', encoding=DEFAULT_ENCODING) as f:
        for line in f:
            _append_new_tokens_(line, tokens=tokens, token_occ=token_occ)

    return tokens, token_occ


def _append_new_tokens_(line, tokens, token_occ):
    i = 0
    while i < len(line):
        tkn, i = _next_token_(line, i)
        if tkn == '':
            break

        if tkn not in tokens:
            tokens.add(tkn)
            token_occ[tkn] = 0

        token_occ[tkn] += 1


def _next_token_(line, i):
    if i == len(line):
        return '', i

    tkn = ''
    if line[i].isalnum():
        while i < len(line) and (line[i].isalnum() or line[i] == '_'):
            tkn += line[i]
            i += 1
        return tkn, i

    # special character
    return line[i], i + 1


if __name__ == '__main__':
    get_and_write_tokens(
        path=os.path.join(HOME_DIR, 'data_dir/kernel_concat/train.txt'),
        limit_occur=3,
        output_file=os.path.join(os.getcwd(), 'tokens.txt')
    )

    # tensor = tokenize_file(
    #     path='/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/data_dir/kernel_concat/train.txt',
    #     tokens_path=os.path.join(os.getcwd(), 'tokens.txt')
    # )
    #
    # print(tensor)
