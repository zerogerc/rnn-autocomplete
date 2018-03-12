import os

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.experiments.linux.batcher import BatcherDataGenerator
from zerogercrnn.experiments.linux.constants import HOME_DIR
from zerogercrnn.experiments.linux.token_level.data import read_data, read_data_mini
from zerogercrnn.experiments.linux.token_level.gru_model import GRULinuxNetwork
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.train.run import TrainEpochRunner
from zerogercrnn.lib.train.routines import BaseRoutine

# --- parameters of the training --- #
config = Config()
config.learning_rate = 5e-3
config.seq_len = 50
config.batch_size = 80
config.embedding_size = 300
config.hidden_size = 1000
config.num_layers = 1
config.dropout = 0.02
config.weight_decay = 0.01
config.epochs = 50
config.decay_after_epoch = 0
config.loss_function = 'nll'
config.optimizer = 'adam'
config.network_type = 'gru'
# ---------------------------------- #

PARAMETERS_PATH = os.path.join(os.getcwd(), 'parameters.json')
TOKENS_PATH = os.path.join(os.getcwd(), 'tokens.txt')
DATA_PATH = os.path.join(HOME_DIR, 'data_dir/kernel_concat')


def create_network(cfg, vocab_size: int):
    if cfg.network_type == 'gru':
        return GRULinuxNetwork(
            vocab_size=vocab_size,
            embedding_size=cfg.embedding_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        )
    else:
        raise Exception('Unknown network type')


def create_loss_function(cfg):
    if cfg.loss_function == 'nll':
        criterion = nn.NLLLoss()
    else:
        raise Exception('Unknown loss function')

    def calc_loss(output_tensor, target_tensor):
        sz_o = output_tensor.size()[-1]
        return criterion(output_tensor.view(-1, sz_o), target_tensor.view(-1))

    return calc_loss


def create_optimizer(cfg, network):
    if cfg.optimizer == 'adam':
        return optim.Adam(params=network.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer')


def create_scheduler(cfg, optimizer):
    return MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(cfg.decay_after_epoch, cfg.epochs + 1)),
        gamma=0.95
    )


def run_train(cfg):
    batcher, corpus = read_data_mini(
        single=HOME_DIR + '/data_dir/linux_kernel_mini.txt',
        tokens_path=TOKENS_PATH,
        seq_len=cfg.seq_len
    )
    # batcher, corpus = read_data(
    #     datadir=DATA_PATH,
    #     tokens_path=TOKENS_PATH,
    #     seq_len=cfg.seq_len
    # )

    data_generator = BatcherDataGenerator(
        batcher=batcher,
        batch_size=cfg.batch_size
    )

    network = create_network(cfg, vocab_size=len(corpus.tokens))
    optimizer = create_optimizer(cfg, network=network)
    scheduler = create_scheduler(cfg, optimizer=optimizer)
    loss_calc = create_loss_function(cfg)

    cfg.write_to_file(PARAMETERS_PATH)

    # load_if_saved(network, path=os.path.join(os.getcwd(), 'saved_models/model_epoch_10'))

    train_routine = BaseRoutine(
        network=network,
        criterion=loss_calc,
        optimizer=optimizer
    )
    validation_routine = BaseRoutine(
        network=network,
        criterion=loss_calc
    )

    runner = TrainEpochRunner(
        network=network,
        train_routine=train_routine,
        validation_routine=validation_routine,
        data_generator=data_generator,
        scheduler=scheduler,
        plotter='tensorboard',
        save_dir=os.path.join(os.getcwd(), 'saved_models')
    )

    runner.run(number_of_epochs=cfg.epochs)


if __name__ == '__main__':
    run_train(cfg=config)
    # parameters = [
    #     'learning_rate',
    #     'weight_decay',
    #     'seq_len',
    #     'batch_size',
    #     'embedding_size',
    #     'hidden_size',
    #     'num_layers',
    #     'dropout',
    #     'weight_decay',
    #     'epochs',
    #     'decay_after_epoch',
    #     'loss_function',
    #     'optimizer'
    # ]
