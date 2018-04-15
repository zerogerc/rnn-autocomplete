import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.experiments.ast_level.main.common import get_optimizer_args, get_scheduler_args
from zerogercrnn.experiments.token_level.data import TokensDataGenerator, TokensDataReader, MockDataReader
from zerogercrnn.experiments.token_level.model import TokenLevelBaseModel
from zerogercrnn.lib.utils.state import load_if_saved
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.train.routines import NetworkRoutine
from zerogercrnn.lib.train.run import TrainEpochRunner
from zerogercrnn.lib.utils.time import logger

parser = argparse.ArgumentParser(description='AST level neural network')
parser.add_argument('--title', type=str, help='Title for this run. Used in tensorboard and in saving of models.')
parser.add_argument('--train_file', type=str, help='File with training data')
parser.add_argument('--eval_file', type=str, help='File with eval data')
parser.add_argument('--embeddings_file', type=str, help='File with embedding vectors')
parser.add_argument('--data_limit', type=int, help='How much lines of data to process (only for fast checking)')
parser.add_argument('--model_save_dir', type=str, help='Where to save trained models')
parser.add_argument('--saved_model', type=str, help='File with trained model if not fresh train')
parser.add_argument('--cuda', action='store_true', help='Use cuda?')
parser.add_argument('--real_data', action='store_true', help='Use real data?')
parser.add_argument('--log', action='store_true', help='Log performance?')
parser.add_argument('--task', type=str, help='One of: train, accuracy')

parser.add_argument('--tokens_count', type=int, help='All possible tokens count')  # 51k now
parser.add_argument('--seq_len', type=int, help='Recurrent layer time unrolling')
parser.add_argument('--batch_size', type=int, help='Size of batch')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--epochs', type=int, help='Number of epochs to run model')
parser.add_argument('--decay_after_epoch', type=int, help='Multiply lr by decay_multiplier each epoch')
parser.add_argument('--decay_multiplier', type=float, help='Multiply lr by this number after decay_after_epoch')
parser.add_argument('--embedding_size', type=int, help='Size of embedding to use')
parser.add_argument('--hidden_size', type=int, help='Hidden size of recurrent part of model')
parser.add_argument('--num_layers', type=int, help='Number of recurrent layers')
parser.add_argument('--dropout', type=float, help='Dropout to apply to recurrent layer')
parser.add_argument('--weight_decay', type=float, help='Weight decay for l2 regularization')

ENCODING = 'ISO-8859-1'


def calc_accuracy(args):
    assert args.eval_file is not None

    data_generator = create_data_generator(args)
    model = create_model(args)
    model.eval()

    if args.saved_model is not None:
        load_if_saved(model, args.saved_model)

    all_tokens = 0
    correct_tokens = 0

    hidden = None
    for iter_data in data_generator.get_eval_generator():
        prediction, target, hidden = run_model(model, iter_data, hidden, args.batch_size, args.cuda)

        _, predicted = torch.max(prediction, dim=2)

        cur_all = target.size()[0] * target.size()[1]
        cur_incorrect = torch.nonzero(target - predicted).size()[0]
        cur_correct = cur_all - cur_incorrect

        all_tokens += cur_all
        cur_all += cur_correct

    print('Accuracy: {}'.format(float(correct_tokens) / all_tokens))


def run_model(model, iter_data, hidden, batch_size, cuda, no_grad):
    (n_input, n_target), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    if no_grad:
        n_input = Variable(n_input, volatile=True)
        n_target = Variable(n_target, volatile=True)
    else:
        n_input = Variable(n_input)
        n_target = Variable(n_target)

    # if cuda:
    #     n_input = n_input.cuda()
    #     n_target = n_target.cuda()

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size, cuda=cuda, no_grad=no_grad)

    model.zero_grad()
    prediction, hidden = model(n_input, hidden, forget_vector=forget_vector)

    return prediction, n_target, hidden


class TokenLevelRoutine(NetworkRoutine):

    def __init__(self, model, batch_size, seq_len, criterion, optimizers, cuda):
        super().__init__(model)
        self.model = self.network  # TODO: refactor base
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.criterion = criterion
        self.optimizers = optimizers
        self.cuda = cuda

        self.hidden = None

    def calc_loss(self, prediction, n_target):
        return self.criterion(prediction.permute(1, 2, 0), n_target.transpose(1, 0))

    def optimize(self, loss):
        # Backward pass
        loss.backward()

        # Optimizer step
        for optimizer in self.optimizers:
            optimizer.step()

    def get_value_from_loss(self, loss):
        return 0

    def run(self, iter_num, iter_data):
        prediction, n_target, hidden = run_model(
            model=self.model,
            iter_data=iter_data,
            hidden=self.hidden,
            batch_size=self.batch_size,
            cuda=self.cuda,
            no_grad=self.optimizers is None
        )
        self.hidden = hidden

        loss = self.calc_loss(prediction, n_target)
        if self.optimizers is not None:
            self.optimize(loss)

        # Return loss value
        return self.get_value_from_loss(loss)


def create_data_generator(args):
    if args.real_data:
        print('Running on real data')
        embeddings = Embeddings(vector_file=args.embeddings_file, embeddings_size=50)

        reader = TokensDataReader(
            train_file=args.train_file,
            eval_file=args.eval_file,
            seq_len=args.seq_len,
            embeddings=embeddings,
            cuda=args.cuda,
            limit=args.data_limit
        )
    else:
        print('Running on mock data')
        reader = MockDataReader(cuda=args.cuda)

    data_generator = TokensDataGenerator(
        data_reader=reader,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        embeddings_size=args.embedding_size,
        cuda=args.cuda
    )

    return data_generator


def create_model(args):
    model = TokenLevelBaseModel(
        embedding_size=args.embedding_size,
        tokens_number=args.tokens_count,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    if args.cuda:
        model = model.cuda()

    return model


def main(args):
    model = create_model(args)

    optimizers = [get_optimizer_args(args, model)]
    schedulers = [get_scheduler_args(args, optimizers[-1])]
    criterion = nn.NLLLoss()

    data_generator = create_data_generator(args)

    train_routine = TokenLevelRoutine(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        criterion=criterion,
        optimizers=optimizers,
        cuda=args.cuda
    )

    validation_routine = TokenLevelRoutine(
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
        data_generator=data_generator,
        schedulers=schedulers,
        plotter='tensorboard',
        save_dir=args.model_save_dir,
        title=args.title
    )

    runner.run(number_of_epochs=args.epochs)


if __name__ == '__main__':
    _args = parser.parse_args()
    assert _args.title is not None
    logger.should_log = _args.log

    if _args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if _args.task == 'train':
        if _args.saved_model is not None:
            raise Exception('Loading of saved model is not supported now')

        main(_args)
    elif _args.task == 'accuracy':
        calc_accuracy(_args)
    else:
        raise Exception('Not supported task')
