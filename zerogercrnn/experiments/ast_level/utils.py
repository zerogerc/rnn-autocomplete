from zerogercrnn.lib.constants import EOF_TOKEN, EMPTY_TOKEN, UNKNOWN_TOKEN, EOF_TOKEN_ID, EMPTY_TOKEN_ID, \
    UNKNOWN_TOKEN_ID
from zerogercrnn.lib.preprocess import read_json

DEFAULT_TERMINALS_FILE = 'data/ast/terminals.json'
DEFAULT_NON_TERMINALS_FILE = 'data/ast/non_terminals.json'


def read_terminals(terminals_file=DEFAULT_TERMINALS_FILE):
    """Returns all terminals in order that they are coded in file_train. """
    terminals = read_json(terminals_file)[:50000 - 1]
    return [EMPTY_TOKEN] + terminals + [UNKNOWN_TOKEN]


def read_non_terminals(non_terminals_file=DEFAULT_NON_TERMINALS_FILE):
    """Returns all non-terminals in order that they are coded in file_train. """
    non_terminals = read_json(non_terminals_file)
    return non_terminals + [EOF_TOKEN]


def get_str2id(strings_array):
    """Returns map from string to index in array. """
    str2id = {}
    for i in range(len(strings_array)):
        str2id[strings_array[i]] = i

    return str2id


if __name__ == '__main__':
    nt = read_non_terminals()
    t = read_terminals()

    nt2id = get_str2id(nt)
    t2id = get_str2id(t)

    assert EOF_TOKEN_ID == nt2id[EOF_TOKEN]
    assert EMPTY_TOKEN_ID == t2id[EMPTY_TOKEN]
    assert UNKNOWN_TOKEN_ID == t2id[UNKNOWN_TOKEN]
