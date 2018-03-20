import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.experiments.js.ast_level.data import ASTDataGenerator, DataReader, MockDataReader
from zerogercrnn.experiments.js.ast_level.network_base_lstm import JSBaseModel
from zerogercrnn.experiments.js.ast_level.train import ASTRoutine
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.train.run import TrainEpochRunner

parser = argparse.ArgumentParser(description='AST level neural network')
parser.add_argument('--config_file', type=str, help='File with training process configuration')
parser.add_argument('--cuda', action='store_true', help='use cuda?')


def create_data_generator(cfg):
    # reader = DataReader(
    #     file_training=cfg.train_file,
    #     file_eval=cfg.eval_file,
    #     encoding=cfg.encoding,
    #     limit_train=cfg.data_train_limit,
    #     limit_eval=cfg.data_eval_limit
    # )

    reader = MockDataReader()

    data_generator = ASTDataGenerator(
        data_reader=reader,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size
    )

    return data_generator


def run_training(cfg, cuda, data_generator, network, criterion, optimizer, scheduler):
    train_routine = ASTRoutine(
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        cuda=cuda
    )

    validation_routine = ASTRoutine(
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        cuda=cuda
    )

    runner = TrainEpochRunner(
        network=network,
        train_routine=train_routine,
        validation_routine=validation_routine,
        data_generator=data_generator,
        scheduler=scheduler,
        plotter='matplotlib',
        save_dir=cfg.model_save_dir
    )

    runner.run(number_of_epochs=cfg.epochs)


def main(cuda, cfg):
    # Data
    data_generator = create_data_generator(cfg)

    # Model
    network = JSBaseModel(
        non_terminal_vocab_size=cfg.non_terminals_count,
        terminal_vocab_size=cfg.terminals_count,
        embedding_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    )

    # Optimizer
    optimizer = optim.Adam(params=network.parameters(), lr=cfg.learning_rate)
    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(cfg.decay_after_epoch, cfg.epochs + 1)),
        gamma=cfg.decay_multiplier
    )

    # Loss function
    base_criterion = nn.NLLLoss()

    def criterion(n_output, n_target):
        """Expect n_output and n_target to be pair of (N, T).
            Return loss as a sum of NLL losses for non-terminal(N) and terminal(T).
        """
        sz_non_terminal = n_output[0].size()[-1]
        # flatten tensors to compute NLLLoss
        loss_non_terminal = base_criterion(
            n_output[0].view(-1, sz_non_terminal),
            n_target[0].view(-1)
        )

        sz_terminal = n_output[1].size()[-1]
        # flatten tensors to compute NLLLoss
        loss_terminal = base_criterion(
            n_output[1].view(-1, sz_terminal),
            n_target[1].view(-1)
        )

        return loss_non_terminal + loss_terminal

    if cuda:
        network = network.cuda()
        base_criterion = base_criterion.cuda()

    # Run
    run_training(cfg, cuda, data_generator, network, criterion, optimizer, scheduler)


if __name__ == '__main__':
    args = parser.parse_args()

    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    config = Config()
    config.read_from_file(args.config_file)

    main(cuda, config)
