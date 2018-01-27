import os
from zerogercrnn.lib.data.character import Corpus, get_alphabet, tokenize_single_without_alphabet

if __name__ == '__main__':
    cwd = os.getcwd()
    print(repr(get_alphabet(cwd, '((character)|(__init__)).py')))
