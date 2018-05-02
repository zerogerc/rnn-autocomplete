import torch
import torch.nn as nn

from zerogercrnn.experiments.ast_level.common import Main, create_non_terminal_embeddings
from zerogercrnn.experiments.ast_level.nt2n_pre.model import NT2NBothTNTPretrainedModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics
from zerogercrnn.lib.run import NetworkRoutine
from zerogercrnn.lib.utils import filter_requires_grad
from zerogercrnn.lib.utils import setup_tensor


def run_model(model, iter_data, hidden, batch_size, cuda, no_grad):
    ((nt_input, t_input), (nt_target, t_target)), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    nt_input = setup_tensor(nt_input, cuda=cuda)
    t_input = setup_tensor(t_input, cuda=cuda)
    nt_target = setup_tensor(nt_target, cuda=cuda)
    t_target = setup_tensor(t_target, cuda=cuda)

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size, cuda=cuda)

    model.zero_grad()
    prediction, hidden = model(nt_input, t_input, hidden, forget_vector=forget_vector)

    return prediction, nt_target, hidden


class ASTRoutine(NetworkRoutine):

    def __init__(self, model, batch_size, seq_len, criterion, optimizers, cuda):
        super().__init__(model)
        self.model = self.network
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.criterion = criterion
        self.optimizers = optimizers
        self.cuda = cuda

        self.hidden = None

    def calc_loss(self, prediction, target):
        return self.criterion(prediction.permute(1, 2, 0), target.transpose(1, 0))

    def optimize(self, loss):
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter_requires_grad(self.model.parameters()), 5)

        # Optimizer step
        for optimizer in self.optimizers:
            optimizer.step()

    def run(self, iter_num, iter_data):
        prediction, target, hidden = run_model(
            model=self.model,
            iter_data=iter_data,
            hidden=self.hidden,
            batch_size=self.batch_size,
            cuda=self.cuda,
            no_grad=self.optimizers is None
        )
        self.hidden = hidden

        if self.optimizers is not None:
            loss = self.calc_loss(prediction, target)
            self.optimize(loss)

        return prediction, target


class NT2NBothTNTPretrainedMain(Main):

    def create_non_terminal_embeddings(self, args):
        return create_non_terminal_embeddings(args)

    def create_model(self, args):
        return NT2NBothTNTPretrainedModel(
            non_terminal_embeddings=self.non_terminal_embeddings,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            prediction_dim=args.non_terminals_num,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    def create_train_routine(self, args):
        return ASTRoutine(
            model=self.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            criterion=self.criterion,
            optimizers=self.optimizers,
            cuda=args.cuda
        )

    def create_validation_routine(self, args):
        return ASTRoutine(
            model=self.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            criterion=self.criterion,
            optimizers=None,
            cuda=args.cuda
        )

    def create_criterion(self, args):
        return nn.CrossEntropyLoss()

    def create_metrics(self, args):
        return MaxPredictionAccuracyMetrics()
