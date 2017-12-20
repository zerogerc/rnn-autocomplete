from __future__ import unicode_literals, print_function, division
import unicodedata
import string
from io import open
import glob

from constants import ROOT_DIR

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


def find_files(pattern):
    """
    Find files in the directory ROOT by pattern.
    :return list of filenames
    """
    return glob.glob('{}/{}'.format(ROOT_DIR, pattern))


def read_lines(filename):
    """
    Read a file and split into lines
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
