import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from lib.data.batcher import Batcher
from lib.data.split import split_rnn_datasets
from lib.utils.file import n_letters
from lib.utils.state import load_if_saved, save_model

from experiments.namegen.networks.classic_lstm import LSTM
from experiments.namegen.reader import names_data_reader
from experiments.namegen.utils import create_train_runner
from experiments.namegen.results import samples

BATCH_SIZE = 128

LEARNING_RATE = 0.001


def train():
    reader = names_data_reader
    input_tensor, target_tensor = reader.get_data()
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_rnn_datasets(input_tensor, target_tensor)

    batcher = Batcher()
    batcher.add_rnn_data('train', train_x, train_y)
    batcher.add_rnn_data('validation', validation_x, validation_y)
    batcher.add_rnn_data('test', test_x, test_y)

    network = LSTM(n_letters + len(reader.all_categories), 128, n_letters)
    load_if_saved(network, path='namegen/lstm')

    optimizer = optim.Adam(network.parameters(), LEARNING_RATE)
    criterion = nn.NLLLoss()

    runner = create_train_runner(network, optimizer, criterion, batcher, BATCH_SIZE)

    all_losses = runner(n_iters=1000, print_every=50, plot_every=5)
    plt.plot(all_losses)
    plt.show()

    save_model(network, path='namegen/lstm')


def sample():
    network = LSTM(n_letters + len(names_data_reader.all_categories), 128, n_letters)
    load_if_saved(network, path='namegen/lstm')

    network.zero_grad()

    samples(network, 'Russian', 'RUS')

    samples(network, 'Arabic', 'ARB')

    samples(network, 'Chinese', 'CHI')

    samples(network, 'Spanish', 'SPA')


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    else:
        sample()
