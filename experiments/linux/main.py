import sys

from experiments.linux.data import data_reader


def train():
    print(data_reader.get_data())


def sample():
    pass


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    else:
        sample()