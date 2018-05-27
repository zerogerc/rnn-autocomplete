from abc import abstractmethod

import torch
import torch.nn as nn

from zerogercrnn.experiments.common import Main
from zerogercrnn.experiments.token_level.data import TokensDataReader, TokensDataGenerator
from zerogercrnn.lib.core import BaseModule
from zerogercrnn.lib.data import BatchedDataGenerator
from zerogercrnn.lib.metrics import Metrics
from zerogercrnn.lib.run import NetworkRoutine
from zerogercrnn.lib.utils import filter_requires_grad, get_best_device


def create_data_generator(args) -> BatchedDataGenerator:
    reader = TokensDataReader(
        train_file=args.train_file,
        eval_file=args.eval_file,
        seq_len=args.seq_len,
        limit=args.data_limit
    )

    data_generator = TokensDataGenerator(
        data_reader=reader,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )

    return data_generator


def run_model(model: nn.Module, iter_data, hidden, batch_size):
    (n_input, n_target), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    n_input = n_input.to(get_best_device())
    n_target = n_target.to(get_best_device())

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size)

    prediction, hidden = model(n_input, hidden, forget_vector=forget_vector)

    return prediction, n_target, hidden


class TokenLevelRoutine(NetworkRoutine):

    def __init__(self, model: nn.Module, batch_size, seq_len, criterion: nn.Module, optimizers):
        super().__init__(model)
        self.model = self.network
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.criterion = criterion
        self.optimizers = optimizers

        self.hidden = None

    def calc_loss(self, prediction, n_target):
        return self.criterion(prediction.permute(1, 2, 0), n_target.transpose(1, 0))  # TODO: check, maybe view?

    def optimize(self, loss):
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter_requires_grad(self.model.parameters()), 5)
        torch.nn.utils.clip_grad_norm_(filter_requires_grad(self.model.sparse_parameters()), 5)

        # Optimizer step
        for optimizer in self.optimizers:
            optimizer.step()

    def run(self, iter_num, iter_data):
        if self.optimizers is not None:
            for optimizer in self.optimizers:
                optimizer.zero_grad()

        prediction, m_target, hidden = run_model(
            model=self.model,
            iter_data=iter_data,
            hidden=self.hidden,
            batch_size=self.batch_size
        )
        self.hidden = hidden

        loss = self.calc_loss(prediction, m_target)
        if self.optimizers is not None:
            self.optimize(loss)

        return prediction, m_target


class TokenMain(Main):

    def __init__(self, args):
        super().__init__(args)

    @abstractmethod
    def create_model(self, args) -> BaseModule:
        pass

    @abstractmethod
    def create_criterion(self, args) -> nn.Module:
        pass

    @abstractmethod
    def create_train_metrics(self, args) -> Metrics:
        pass

    @abstractmethod
    def create_eval_metrics(self, args) -> Metrics:
        pass

    def create_data_generator(self, args) -> BatchedDataGenerator:
        return create_data_generator(args)

    def create_train_routine(self, args) -> NetworkRoutine:
        return TokenLevelRoutine(
            model=self.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            criterion=self.criterion,
            optimizers=self.optimizers
        )

    def create_validation_routine(self, args) -> NetworkRoutine:
        return TokenLevelRoutine(
            model=self.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            criterion=self.criterion,
            optimizers=None
        )
