import os

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.experiments.linux.token_level.data import read_data_mini, read_data
from zerogercrnn.experiments.linux.constants import HOME_DIR
from zerogercrnn.experiments.linux.token_level.gru_model import GRULinuxNetwork
from zerogercrnn.lib.train.run import TrainEpochRunner

# ------------- hyperparameters ------------- #


# Context
SEQ_LEN = 50

# Batch
BATCH_SIZE = 80

# Learning rate
LEARNING_RATE = 3 * 1e-3

# Network parameters
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 1500
NUM_LAYERS = 1
DROPOUT = 0.02
WEIGHT_DECAY = 0.01

# Number of train epoch
EPOCHS = 50

# Decrease learning rate by 0.9 after this epoch
DECAY_AFTER_EPOCH = 0

# ------------------------------------------- #

TOKENS_PATH = os.path.join(os.getcwd(), 'tokens.txt')
DATA_PATH = os.path.join(HOME_DIR, 'data_dir/kernel_concat')

# def create_lstm(input_size, output_size):
#     return LSTMLinuxNetwork(
#         input_size=input_size,
#         hidden_size=HIDDEN_SIZE,
#         output_size=output_size,
#         num_layers=NUM_LAYERS,
#         dropout=DROPOUT
#     )
#
#
# def create_rnn(input_size, output_size):
#     return RNNLinuxNetwork(
#         input_size=input_size,
#         hidden_size=HIDDEN_SIZE,
#         output_size=output_size,
#         num_layers=NUM_LAYERS,
#         dropout=DROPOUT
#     )


def create_gru(vocab_size):
    return GRULinuxNetwork(
        vocab_size=vocab_size,
        embedding_size=200,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )


def run_train():
    # batcher, corpus = read_data_mini(
    #     single=HOME_DIR + '/data_dir/linux_kernel_mini.txt',
    #     tokens_path=TOKENS_PATH,
    #     seq_len=SEQ_LEN
    # )
    batcher, corpus = read_data(
        datadir=DATA_PATH,
        tokens_path=TOKENS_PATH,
        seq_len=SEQ_LEN
    )

    network = create_gru(vocab_size=len(corpus.tokens))

    # load_if_saved(network, path=os.path.join(os.getcwd(), 'saved_models/model_epoch_10'))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
        save_dir=os.path.join(os.getcwd(), 'saved_models')
    )

    runner.run(number_of_epochs=EPOCHS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    run_train()
