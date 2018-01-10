import sys

import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim

from experiments.linux.data import read_data_mini, read_data
from experiments.linux.lstm import LSTMLinuxNetwork
from lib.train.run import TrainEpochRunner

BATCH_SIZE = 100

LEARNING_RATE = 2 * 1e-3

HIDDEN_SIZE = 64
NUM_LAYERS = 1

EPOCHS = 50
DECAY_AFTER_EPOCH = 10


def run_train():
    batcher, corpus = read_data_mini()

    INPUT_SIZE = len(corpus.all_letters)
    OUTPUT_SIZE = len(corpus.all_letters)

    network = LSTMLinuxNetwork(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS
    )
    criterion = nn.NLLLoss()
    # optimizer = optim.Adam(params=network.parameters(), lr=LEARNING_RATE)
    optimizer = optim.RMSprop(params=network.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # We will decay after DECAY_AFTER_EPOCH by WEIGHT_DECAY after each additional epoch
    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(DECAY_AFTER_EPOCH, EPOCHS + 1)),
        gamma=0.95
    )

    def calc_loss(output_tensor, target_tensor):
        sz_o = output_tensor.size()[-1]
        return criterion(output_tensor.view(-1, sz_o), target_tensor.view(-1))

    runner = TrainEpochRunner(
        network=network,
        loss_calc=calc_loss,
        optimizer=optimizer,
        batcher=batcher,
        scheduler=scheduler
    )

    runner.run(number_of_epochs=EPOCHS, batch_size=BATCH_SIZE)


def run_sample():
    pass


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        run_train()
    else:
        run_sample()
