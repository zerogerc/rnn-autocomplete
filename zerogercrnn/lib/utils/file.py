from __future__ import unicode_literals, print_function, division

import os
import glob
import string
import unicodedata
from io import open

import numpy as np

from zerogercrnn.lib.utils.split import get_split_indexes

DEFAULT_ENCODING = 'ISO-8859-1'

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


def find_files(root, patterns):
    """
    Find files in the directory ROOT by pattern.
    :return list of filenames
    """
    files = []
    for pattern in patterns:
        files.extend(glob.glob('{}/{}'.format(root, pattern), recursive=True))
    return files


def read_lines(filename, encoding):
    """
    Read a file and split into lines
    """
    """Read first *limit* lines from file and returns list of them."""
    lines = []
    with open(filename, mode='r', encoding=encoding) as f:
        for l in f:
            if l[-1] == '\n':
                lines.append(l[:-1])
            else:
                lines.append(l)
    return lines


def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def convert_repo_to_train_val_test(root, patterns, dest_dir, validation_percentage, test_percentage):
    """Find files in repo that match pattern and concatenate them 
    into three different files for train, validation and test.
    
    :param root: home directory of the project. All paths will be relative to this.
    :param pattern: array of patterns to find files by.
    :param dest_dir: directory to store files to. Files will be **train.txt**, **validation.txt**, **test.txt**.
    :param validation_percentage: percentage of validation data.
    :param test_percentage: percentage of test data.
    """
    filenames = np.array(find_files(root, patterns))

    train_file = open('{}{}/{}'.format(root, dest_dir, 'train.txt'), 'w', encoding=DEFAULT_ENCODING)
    validation_file = open('{}{}/{}'.format(root, dest_dir, 'validation.txt'), 'w', encoding=DEFAULT_ENCODING)
    test_file = open('{}{}/{}'.format(root, dest_dir, 'test.txt'), 'w', encoding=DEFAULT_ENCODING)

    train_indexes, validation_indexes, test_indexes = get_split_indexes(
        len(filenames),
        validation_percentage,
        test_percentage,
        shuffle=True
    )

    for train_i in train_indexes:
        train_file.write(open(filenames[train_i], 'r', encoding=DEFAULT_ENCODING).read())

    for validation_i in validation_indexes:
        validation_file.write(open(filenames[validation_i], 'r', encoding=DEFAULT_ENCODING).read())

    for test_i in test_indexes:
        test_file.write(open(filenames[test_i], 'r', encoding=DEFAULT_ENCODING).read())


if __name__ == '__main__':
    convert_repo_to_train_val_test('data/kernel/*.[c|h]', '/data/kernel_concatenated', 0.05, 0.05)
