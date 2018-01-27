import sys
import os

import torch.nn as nn
import torch.optim as optim
from zerogercrnn.experiments.linux.lstm import LSTMLinuxNetwork
from zerogercrnn.lib.train.run import TrainEpochRunner
from zerogercrnn.lib.utils.state import save_model
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.experiments.linux.data import read_data, read_data_mini

SEQ_LEN = 100
BATCH_SIZE = 64

LEARNING_RATE = 5 * 1e-3

HIDDEN_SIZE = 128
NUM_LAYERS = 1

EPOCHS = 50
DECAY_AFTER_EPOCH = 10


def run_train():
    batcher, corpus = read_data_mini(single=os.getcwd() + '/data/linux_kernel_mini.txt', seq_len=SEQ_LEN)

    INPUT_SIZE = len(corpus.alphabet)
    OUTPUT_SIZE = len(corpus.alphabet)

    network = LSTMLinuxNetwork(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS
    )
    criterion = nn.NLLLoss()
    # optimizer = optim.Adam(params=network.parameters(), lr=LEARNING_RATE)
    optimizer = optim.RMSprop(params=network.parameters(), lr=LEARNING_RATE)

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
    save_model(network, 'linux')


def run_sample():
    pass


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        run_train()
    else:
        run_sample()
