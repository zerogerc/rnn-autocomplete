import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.experiments.argutils import add_general_arguments, add_batching_data_args, add_optimization_args, \
    add_recurrent_core_args, add_non_terminal_args, add_terminal_args
from zerogercrnn.experiments.common import get_optimizer_args, get_scheduler_args
from zerogercrnn.lib.metrics import AccuracyMetrics
from zerogercrnn.lib.utils.state import load_cuda_on_cpu, load_if_saved

from zerogercrnn.lib.train.run import TrainEpochRunner
from zerogercrnn.lib.train.routines import NetworkRoutine
from zerogercrnn.experiments.ast_level.nt2n.data import ASTDataReader, ASTDataGenerator
from zerogercrnn.experiments.ast_level.nt2n.model import NT2NBaseModel
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils.time import logger

parser = argparse.ArgumentParser(description='AST level neural network')
add_general_arguments(parser)
add_batching_data_args(parser)
add_optimization_args(parser)
add_recurrent_core_args(parser)
add_non_terminal_args(parser)
add_terminal_args(parser)
parser.add_argument('--terminal_embeddings_file', type=str, help='File with pretrained terminal embeddings')


def wrap_cuda_no_grad_variable(tensor, cuda, no_grad=False):
    wrapped = None
    if no_grad:
        wrapped = Variable(tensor, volatile=True)
    else:
        wrapped = Variable(tensor)

    if cuda:
        wrapped = wrapped.cuda()
    return wrapped


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

        loss = self.calc_loss(prediction, target)
        if self.optimizers is not None:
            self.optimize(loss)

        return prediction, target


def create_terminal_embeddings(args):
    return Embeddings(embeddings_size=args.terminal_embedding_dim, vector_file=args.terminal_embeddings_file, squeeze=True)


def create_data_generator(args):
    data_reader = ASTDataReader(
        file_train=args.train_file,
        file_eval=args.eval_file,
        cuda=args.cuda,
        seq_len=args.seq_len,
        number_of_seq=20,
        limit=args.data_limit
    )

    data_generator = ASTDataGenerator(
        data_reader=data_reader,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        cuda=args.cuda
    )

    return data_generator


def create_model(args, terminal_embeddings):
    model = NT2NBaseModel(
        non_terminals_num=args.non_terminals_num,
        non_terminal_embedding_dim=args.non_terminal_embedding_dim,
        terminal_embeddings=terminal_embeddings,
        hidden_dim=args.hidden_size,
        prediction_dim=args.non_terminals_num,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    if args.cuda:
        model = model.cuda()

    return model


def train(args):
    terminal_embeddings = create_terminal_embeddings(args)
    model = create_model(args, terminal_embeddings)
    optimizers = [get_optimizer_args(args, model)]
    schedulers = [get_scheduler_args(args, optimizers[-1])]
    criterion = nn.NLLLoss()

    if args.saved_model is not None:
        if args.cuda:
            load_if_saved(model, args.saved_model)
        else:
            load_cuda_on_cpu(model, args.saved_model)

    data_generator = create_data_generator(args)

    train_routine = ASTRoutine(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        criterion=criterion,
        optimizers=optimizers,
        cuda=args.cuda
    )

    validation_routine = ASTRoutine(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        criterion=criterion,
        optimizers=None,
        cuda=args.cuda
    )

    runner = TrainEpochRunner(
        network=model,
        train_routine=train_routine,
        validation_routine=validation_routine,
        metrics=AccuracyMetrics(),
        data_generator=data_generator,
        schedulers=schedulers,
        plotter='tensorboard',
        save_dir=args.model_save_dir,
        title=args.title,
        plot_train_every=50
    )

    runner.run(number_of_epochs=args.epochs)


if __name__ == '__main__':
    _args = parser.parse_args()
    assert _args.title is not None
    logger.should_log = _args.log

    if _args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    train(_args)
