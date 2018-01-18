import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from experiments.trends.trends_gru import TrendsGRU
from experiments.trends.trends_rnn import TrendsRNN
from lib.old.batcher import Batcher
from lib.old.sample import RNNSampler
from lib.train.run import TrainEpochRunner
from lib.utils.split import split_rnn_datasets
from lib.utils.state import save_model, load_if_saved
from torch.autograd import Variable

from zerogerc_rnn.experiments.trends.data import data_reader

BATCH_SIZE = 8
LEARNING_RATE = 0.001


def loss_calculator(output_tensor, target_tensor, criterion, seq_len):
    sz_o = output_tensor.size()[-1]
    sz_t = target_tensor.size()[-1]

    loss = criterion(output_tensor.view(-1, sz_o), target_tensor.view(-1, sz_t))
    return loss


def read_data():
    input_tensor, target_tensor = data_reader.get_data()
    input_tensor = input_tensor / 100
    target_tensor = target_tensor / 100

    train_x, train_y, validation_x, validation_y, test_x, test_y \
        = split_rnn_datasets(input_tensor, target_tensor, validation_percentage=0.1, test_percentage=0.1, shuffle=False)

    batcher = Batcher()
    batcher.add_rnn_data('train', train_x, train_y)
    batcher.add_rnn_data('validation', validation_x, validation_y)
    batcher.add_rnn_data('test', test_x, test_y)

    return batcher

def train():
    batcher = read_data()

    network = TrendsGRU(
        input_size=1,  # single number
        hidden_size=32,
        output_size=1  # single number
    )

    # network = TrendsRNN(
    #     input_size=1,
    #     hidden_size=32,
    #     output_size=1
    # )

    load_if_saved(network, path='trends/rnn')

    optimizer = optim.Adam(network.parameters(), LEARNING_RATE)
    criterion = nn.MSELoss()

    runner = TrainEpochRunner(
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        batcher=batcher
    )

    runner.run(
        number_of_epochs=100,
        batch_size=8
    )

    save_model(network, path='trends/rnn')


def sample():
    input_tensor, target_tensor = data_reader.get_data()
    input_tensor = input_tensor / 100
    target_tensor = target_tensor / 100

    network = TrendsRNN(
        input_size=1,  # single number
        hidden_size=32,
        output_size=1  # single number
    )
    load_if_saved(network, path='trends/rnn')

    sampler = RNNSampler(network)

    # initial input (first 12 numbers)
    initial_input = Variable(input_tensor[:, :1, :].float())
    # sample others
    sampled = sampler.sample(initial_input, input_tensor.size()[1])
    # concat initial and sampled
    sampled = torch.cat((initial_input.view(data_reader.SEQ_LEN, -1), sampled), dim=0).view(-1)

    # draw true data
    plt.plot(data_reader.row['count'] / 100, label='True')
    # draw sampled data
    plt.plot(sampled.data.numpy(), label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    else:
        sample()
