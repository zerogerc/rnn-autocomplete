import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from lib.data.split import split_rnn_datasets
from lib.data.batcher import Batcher
from lib.utils.state import save_model, load_if_saved
from lib.utils.train import RNNRunner
from lib.utils.sample import RNNSampler

from experiments.trends.data import data_reader
from experiments.trends.trends_gru import TrendsGRU

BATCH_SIZE = 16
LEARNING_RATE = 0.001


def loss_calculator(output_tensor, target_tensor, criterion, seq_len):
    loss = 0
    for i in range(seq_len):
        loss += criterion(output_tensor[i], target_tensor[i])
    return loss / seq_len


def train():
    input_tensor, target_tensor = data_reader.get_data()
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_rnn_datasets(input_tensor, target_tensor)

    batcher = Batcher()
    batcher.add_rnn_data('train', train_x, train_y)
    batcher.add_rnn_data('validation', validation_x, validation_y)
    batcher.add_rnn_data('test', test_x, test_y)

    network = TrendsGRU(
        input_size=1,  # single number
        hidden_size=32,
        output_size=1  # single number
    )
    load_if_saved(network, path='trends/gru')

    optimizer = optim.Adam(network.parameters(), LEARNING_RATE)
    criterion = nn.MSELoss()

    runner = RNNRunner(
        network=network,
        optimizer=optimizer,
        criterion=criterion,
        batcher=batcher,
        seq_len=data_reader.SEQ_LEN,
        loss_calculator=loss_calculator
    )

    runner.run_train(
        batch_size=BATCH_SIZE,
        n_iters=1000,
        validation_every=100,
        print_every=100
    )

    save_model(network, path='trends/gru')


def sample():
    input_tensor, target_tensor = data_reader.get_data()

    network = TrendsGRU(
        input_size=1,  # single number
        hidden_size=32,
        output_size=1  # single number
    )
    load_if_saved(network, path='trends/gru')

    sampler = RNNSampler(network)

    # initial input (first 12 numbers)
    initial_input = Variable(input_tensor[:, :1, :].float())
    # sample others
    sampled = sampler.sample(initial_input, input_tensor.size()[1] - data_reader.SEQ_LEN)
    # concat initial and sampled
    sampled = torch.cat((initial_input.view(data_reader.SEQ_LEN, -1), sampled), dim=0).view(-1)

    # draw true data
    plt.plot(data_reader.row['count'], label='True')
    # draw sampled data
    plt.plot(sampled.data.numpy(), label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    else:
        sample()
