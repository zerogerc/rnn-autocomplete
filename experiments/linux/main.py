import sys

from constants import ROOT_DIR
from lib.data.text_reader import RowTextDataReader
from lib.data.batcher import Batcher
from lib.data.split import split_data

DATA_PATH = ROOT_DIR + '/data/linux_kernel_mini.txt'
SEQ_LEN = 100

def run_train():
    reader = RowTextDataReader(DATA_PATH)
    input_tensor = reader.get_data()
    print(input_tensor.size())

    train, validation, test = split_data(input_tensor, validation_percentage=0.1, test_percentage=0.1, shuffle=False)
    batcher = Batcher()
    batcher.add_rnn_data_generator('train', train, SEQ_LEN)
    batcher.add_rnn_data_generator('validation', validation, SEQ_LEN)
    batcher.add_rnn_data_generator('test', test, SEQ_LEN)

    print(train.size())
    print(validation.size())
    print(test.size())


def run_sample():
    pass


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        run_train()
    else:
        run_sample()