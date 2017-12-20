import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from lib.data.split import split_datasets
from lib.data.batcher import Batcher
from lib.utils.file import n_letters

from experiments.namegen.networks.SimpleRNN import RNN
from experiments.namegen.reader import names_data_reader
from experiments.namegen.utils import create_train_runner

BATCH_SIZE = 128

LEARNING_RATE = 0.0001


def main():
    reader = names_data_reader
    input_tensor, target_tensor = reader.get_data()

    train_x, train_y, validation_x, validation_y, test_x, test_y = split_datasets(input_tensor, target_tensor)

    batcher = Batcher()
    batcher.add_rnn_data('train', train_x, train_y)
    batcher.add_rnn_data('validation', validation_x, validation_y)
    batcher.add_rnn_data('test', test_x, test_y)

    network = RNN(n_letters + len(reader.all_categories), 128, n_letters)

    optimizer = optim.Adam(network.parameters(), LEARNING_RATE)
    criterion = nn.NLLLoss()

    runner = create_train_runner(network, optimizer, criterion, batcher, BATCH_SIZE)

    all_losses = runner(100, 5, 1)
    plt.plot(all_losses)
    plt.show()


if __name__ == '__main__':
    main()
