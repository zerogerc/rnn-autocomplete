import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.experiments.js.ast_level.data import ASTDataGenerator, DataReader, MockDataReader
from zerogercrnn.experiments.js.ast_level.network_base import JSBaseModel, RecurrentCore
from zerogercrnn.experiments.js.ast_level.train_routine import ASTRoutine
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.train.run import TrainEpochRunner
from zerogercrnn.lib.utils.time import logger

parser = argparse.ArgumentParser(description='AST level neural network')
parser.add_argument('--config_file', type=str, help='File with training process configuration')
parser.add_argument('--title', type=str, help='Title for this run')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--real_data', action='store_true', help='use real data?')
parser.add_argument('--log', action='store_true', help='log performance?')

"""
File to be able to train model from console. You could specify params of model in config.json file.
The location of this file is command line parameter 
"""


def create_data_generator(cfg, real_data):
    """Create DataReader with either real or fake data."""
    if real_data:
        reader = DataReader(
            file_training=cfg.train_file,
            file_eval=cfg.eval_file,
            encoding=cfg.encoding,
            limit_train=cfg.data_train_limit,
            limit_eval=cfg.data_eval_limit,
            cuda=True
        )
    else:
        reader = MockDataReader()

    data_generator = ASTDataGenerator(
        data_reader=reader,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size
    )

    return data_generator


def run_training(cfg, title, cuda, data_generator, network, criterion, optimizers, schedulers):
    train_routine = ASTRoutine(
        network=network,
        criterion=criterion,
        optimizers=optimizers,
        cuda=cuda
    )

    validation_routine = ASTRoutine(
        network=network,
        criterion=criterion,
        cuda=cuda
    )

    runner = TrainEpochRunner(
        network=network,
        train_routine=train_routine,
        validation_routine=validation_routine,
        data_generator=data_generator,
        schedulers=schedulers,
        plotter='tensorboard',
        title=title,
        save_dir=cfg.model_save_dir
    )

    runner.run(number_of_epochs=cfg.epochs)


def main(title, cuda, real_data, cfg):
    # Data
    data_generator = create_data_generator(cfg, real_data)

    # Model
    recurrent_code = RecurrentCore(
        input_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        model_type='gru'
    )
    if cuda:
        recurrent_code = recurrent_code.cuda()

    recurrent_code.init_hidden(cfg.batch_size, cuda)

    model = JSBaseModel(
        non_terminal_vocab_size=cfg.non_terminals_count,
        terminal_vocab_size=cfg.terminals_count,
        embedding_size=cfg.embedding_size,
        recurrent_layer=recurrent_code
    )

    if cuda:
        model = model.cuda()

    # Optimizer
    dense_optimizer = optim.Adam(params=model.dense_params)
    sparse_optimizer = optim.SparseAdam(params=model.sparse_params)
    # optimizer = optim.SGD(params=model.parameters(), lr=cfg.learning_rate)

    dense_scheduler = MultiStepLR(
        optimizer=dense_optimizer,
        milestones=list(range(cfg.decay_after_epoch, cfg.epochs + 1)),
        gamma=cfg.decay_multiplier
    )
    sparse_scheduler = MultiStepLR(
        optimizer=sparse_optimizer,
        milestones=list(range(cfg.decay_after_epoch, cfg.epochs + 1)),
        gamma=cfg.decay_multiplier
    )

    # Loss function
    base_criterion = nn.NLLLoss()
    if cuda:
        base_criterion = base_criterion.cuda()

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

    # Run
    run_training(
        cfg=cfg,
        title=title,
        cuda=cuda,
        data_generator=data_generator,
        network=model,
        criterion=criterion,
        optimizers=[dense_optimizer, sparse_optimizer],
        schedulers=[dense_scheduler, sparse_scheduler]
    )


if __name__ == '__main__':
    args = parser.parse_args()

    assert args.title is not None
    cuda = args.cuda
    logger.should_log = args.log
    real_data = args.real_data

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    config = Config()
    config.read_from_file(args.config_file)

    main(args.title, cuda, real_data, config)
