import torch
import torch.nn as nn

from zerogercrnn.experiments.ast_level.common import ASTMain
from zerogercrnn.experiments.ast_level.nt2n_cross.model import NT2NBaseModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics
from zerogercrnn.lib.run import NetworkRoutine
from zerogercrnn.lib.utils import filter_requires_grad
from zerogercrnn.lib.utils import setup_tensor


def run_model(model, iter_data, hidden, batch_size):
    ((nt_input, t_input), (nt_target, t_target)), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    nt_input = setup_tensor(nt_input)
    t_input = setup_tensor(t_input)
    nt_target = setup_tensor(nt_target)
    t_target = setup_tensor(t_target)

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size)

    model.zero_grad()
    prediction, hidden = model(nt_input, t_input, hidden, forget_vector=forget_vector)

    return prediction, nt_target, hidden


class ASTRoutine(NetworkRoutine):

    def __init__(self, model, batch_size, seq_len, criterion, optimizers):
        super().__init__(model)
        self.model = self.network
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.criterion = criterion
        self.optimizers = optimizers

        self.hidden = None

    def calc_loss(self, prediction, target):
        return self.criterion(prediction.view(-1, prediction.size()[-1]), target.view(-1))

    def optimize(self, loss):
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter_requires_grad(self.model.parameters()), 5)

        # Optimizer step
        for optimizer in self.optimizers:
            optimizer.step()

    def run(self, iter_num, iter_data):
        prediction, target, hidden = run_model(model=self.model, iter_data=iter_data, hidden=self.hidden,
                                               batch_size=self.batch_size)
        self.hidden = hidden

        if self.optimizers is not None:
            loss = self.calc_loss(prediction, target)
            self.optimize(loss)

        return prediction, target


class NT2NCrossMain(ASTMain):
    def create_model(self, args):
        return NT2NBaseModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            prediction_dim=args.non_terminals_num,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    def create_train_routine(self, args):
        return ASTRoutine(model=self.model, batch_size=args.batch_size, seq_len=args.seq_len, criterion=self.criterion,
                          optimizers=self.optimizers)

    def create_validation_routine(self, args):
        return ASTRoutine(model=self.model, batch_size=args.batch_size, seq_len=args.seq_len, criterion=self.criterion,
                          optimizers=None)

    def create_criterion(self, args):
        return nn.CrossEntropyLoss()

    def create_metrics(self, args):
        return MaxPredictionAccuracyMetrics()
