import torch

from zerogercrnn.experiments.ast_level.attention_nt2n.model import NT2NAttentionModel
from zerogercrnn.experiments.ast_level.common import Main
from zerogercrnn.experiments.ast_level.nt2n.model import NT2NBaseModel
from zerogercrnn.experiments.utils import filter_requires_grad
from zerogercrnn.experiments.utils import wrap_cuda_no_grad_variable
from zerogercrnn.lib.metrics import AccuracyMetrics
from zerogercrnn.lib.run import NetworkRoutine


def run_model(model, iter_data, hidden, batch_size, cuda, no_grad):
    ((nt_input, t_input), (nt_target, t_target)), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    nt_input = wrap_cuda_no_grad_variable(nt_input, cuda=cuda, no_grad=no_grad)
    t_input = wrap_cuda_no_grad_variable(t_input, cuda=cuda, no_grad=no_grad)
    nt_target = wrap_cuda_no_grad_variable(nt_target, cuda=cuda, no_grad=no_grad)
    t_target = wrap_cuda_no_grad_variable(t_target, cuda=cuda, no_grad=no_grad)

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size, cuda=cuda, no_grad=no_grad)

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
        # torch.nn.utils.clip_grad_norm(filter_requires_grad(self.model.parameters()), 5)

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


class NT2NMain(Main):

    def __init__(self, args, attention=False):
        self.attention = attention
        super().__init__(args)

    def create_model(self, args):
        if self.attention:
            return NT2NAttentionModel(
                seq_len=args.seq_len - 1,
                non_terminals_num=args.non_terminals_num,
                non_terminal_embedding_dim=args.non_terminal_embedding_dim,
                terminal_embeddings=self.terminal_embeddings,
                hidden_dim=args.hidden_size,
                prediction_dim=args.non_terminals_num,
                dropout=args.dropout
            )
        else:
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

    def create_metrics(self, args):
        return AccuracyMetrics()
