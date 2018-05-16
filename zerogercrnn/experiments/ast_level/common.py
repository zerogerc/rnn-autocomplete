from abc import abstractmethod

import torch
import torch.nn as nn

from zerogercrnn.experiments.ast_level.data import ASTInput, ASTTarget, ASTDataReader, ASTDataGenerator
from zerogercrnn.experiments.common import Main
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.metrics import Metrics
from zerogercrnn.lib.run import NetworkRoutine
from zerogercrnn.lib.utils import filter_requires_grad


# region Utils

def create_terminal_embeddings(args):
    return Embeddings(
        embeddings_size=args.terminal_embedding_dim,
        vector_file=args.terminal_embeddings_file,
        squeeze=True
    )


def create_non_terminal_embeddings(args):
    return Embeddings(
        embeddings_size=args.non_terminal_embedding_dim,
        vector_file=args.non_terminal_embeddings_file,
        squeeze=False
    )


def create_data_generator(args):
    data_reader = ASTDataReader(
        file_train=args.train_file,
        file_eval=args.eval_file,
        seq_len=args.seq_len,
        number_of_seq=20,
        limit=args.data_limit
    )

    data_generator = ASTDataGenerator(
        data_reader=data_reader,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )

    return data_generator


# endregion

# region Loss

class ASTLoss(nn.Module):

    @abstractmethod
    def forward(self, prediction: torch.Tensor, target: ASTTarget):
        pass


class NonTerminalsCrossEntropyLoss(ASTLoss):

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, prediction: torch.Tensor, target: ASTTarget):
        return self.criterion(prediction.view(-1, prediction.size()[-1]), target.non_terminals.view(-1))


class TerminalsCrossEntropyLoss(ASTLoss):

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, prediction: torch.Tensor, target: ASTTarget):
        return self.criterion(prediction.view(-1, prediction.size()[-1]), target.terminals.view(-1))


# endregion

# region Metrics

class NonTerminalMetrics(Metrics):

    def __init__(self, base: Metrics):
        super().__init__()
        self.base = base

    def drop_state(self):
        self.base.drop_state()

    def report(self, prediction_target):
        prediction, target = prediction_target
        self.base.report((prediction, target.non_terminals))

    def get_current_value(self, should_print=False):
        return self.base.get_current_value(should_print=should_print)


class TerminalMetrics(Metrics):

    def __init__(self, base: Metrics):
        super().__init__()
        self.base = base

    def drop_state(self):
        self.base.drop_state()

    def report(self, prediction_target):
        prediction, target = prediction_target
        self.base.report((prediction, target.terminals))

    def get_current_value(self, should_print=False):
        return self.base.get_current_value(should_print=should_print)


# endregion

# region Routine

def run_model(model, iter_data, hidden, batch_size):
    (m_input, m_target), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    m_input = ASTInput.setup(m_input)
    m_target = ASTTarget.setup(m_target)

    m_input.current_non_terminals = m_target.non_terminals

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size)

    prediction, hidden = model(m_input, hidden, forget_vector=forget_vector)

    return prediction, m_target, hidden


class ASTRoutine(NetworkRoutine):

    def __init__(self, model, batch_size, seq_len, criterion: ASTLoss, optimizers):
        super().__init__(model)
        self.model = self.network
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.criterion = criterion
        self.optimizers = optimizers

        self.hidden = None

    def optimize(self, loss):
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter_requires_grad(self.model.parameters()), 5)
        # torch.nn.utils.clip_grad_norm_(filter_requires_grad(self.model.sparse_parameters()), 5)

        # Optimizer step
        for optimizer in self.optimizers:
            optimizer.step()

    def run(self, iter_num, iter_data):
        if self.optimizers is not None:
            for optimizer in self.optimizers:
                optimizer.zero_grad()

        prediction, target, hidden = run_model(
            model=self.model,
            iter_data=iter_data,
            hidden=self.hidden,
            batch_size=self.batch_size
        )
        self.hidden = hidden

        if self.optimizers is not None:
            loss = self.criterion(prediction, target)
            self.optimize(loss)

        return prediction, target


# endregion

class ASTMain(Main):
    def __init__(self, args):
        self.non_terminal_embeddings = self.create_non_terminal_embeddings(args)
        self.terminal_embeddings = self.create_terminal_embeddings(args)
        super().__init__(args)

    @abstractmethod
    def create_model(self, args):
        pass

    @abstractmethod
    def create_criterion(self, args):
        pass

    @abstractmethod
    def create_metrics(self, args):
        pass

    def create_data_generator(self, args):
        return create_data_generator(args)

    def create_terminal_embeddings(self, args):
        return None

    def create_non_terminal_embeddings(self, args):
        return None

    def create_train_routine(self, args):
        return ASTRoutine(
            model=self.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            criterion=self.criterion,
            optimizers=self.optimizers
        )

    def create_validation_routine(self, args):
        return ASTRoutine(
            model=self.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            criterion=self.criterion,
            optimizers=None
        )
