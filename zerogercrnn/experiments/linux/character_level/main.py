import os

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.experiments.linux.character_level.data import read_data
from zerogercrnn.experiments.linux.models.gru import GRULinuxNetwork
from zerogercrnn.experiments.linux.models.lstm import LSTMLinuxNetwork
from zerogercrnn.experiments.linux.models.rnn import RNNLinuxNetwork
from zerogercrnn.lib.train.run import TrainEpochRunner

# ------------- hyperparameters ------------- #


# Context
SEQ_LEN = 100

# Batch
BATCH_SIZE = 100

# Learning rate
LEARNING_RATE = 2 * 1e-3

# Network parameters
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.02

# Number of train epoch
EPOCHS = 50

# Decrease learning rate by 0.9 after this epoch
DECAY_AFTER_EPOCH = 10


# ------------------------------------------- #



def create_lstm(input_size, output_size):
    return LSTMLinuxNetwork(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=output_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )


def create_rnn(input_size, output_size):
    return RNNLinuxNetwork(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=output_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )


def create_gru(input_size, output_size):
    return GRULinuxNetwork(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=output_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )


def run_train():
    # batcher, corpus = read_data_mini(single=os.getcwd() + '/data_dir/linux_kernel_mini.txt', seq_len=SEQ_LEN)
    batcher, corpus = read_data(datadir=os.path.join(os.getcwd(), 'data_dir/kernel_concat/'), seq_len=SEQ_LEN)

    INPUT_SIZE = len(corpus.alphabet)
    OUTPUT_SIZE = len(corpus.alphabet)

    network = create_gru(INPUT_SIZE, OUTPUT_SIZE)

    # load_if_saved(network, path=os.path.join(os.getcwd(), 'saved_models/model_epoch_10'))

    criterion = nn.NLLLoss()
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
        scheduler=scheduler,
        plotter='visdom',
        # save_dir=os.path.join(os.getcwd(), 'saved_models')
    )

    runner.run(number_of_epochs=EPOCHS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    run_train()
