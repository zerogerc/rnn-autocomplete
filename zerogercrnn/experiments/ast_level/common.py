from abc import abstractmethod
from torch import nn as nn

from zerogercrnn.experiments.ast_level.data import ASTDataReader, ASTDataGenerator
from zerogercrnn.experiments.common import get_optimizers, get_scheduler_args
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.file import load_if_saved, load_cuda_on_cpu
from zerogercrnn.lib.run import TrainEpochRunner


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


class Main:
    def __init__(self, args):
        self.non_terminal_embeddings = self.create_non_terminal_embeddings(args)
        self.terminal_embeddings = self.create_terminal_embeddings(args)

        self.model = self.create_model(args)
        self.load_model(args)

        if args.cuda:
            self.model = self.model.cuda()

        self.optimizers = self.create_optimizers(args)
        self.schedulers = self.create_schedulers(args)
        self.criterion = self.create_criterion(args)

        self.data_generator = self.create_data_generator(args)

        self.train_routine = self.create_train_routine(args)
        self.validation_routine = self.create_validation_routine(args)
        self.metrics = self.create_metrics(args)
        self.plotter = 'tensorboard'

    @abstractmethod
    def create_model(self, args):
        pass

    @abstractmethod
    def create_train_routine(self, args):
        pass

    @abstractmethod
    def create_validation_routine(self, args):
        pass

    @abstractmethod
    def create_metrics(self, args):
        pass

    def train(self, args):
        runner = TrainEpochRunner(
            network=self.model,
            train_routine=self.train_routine,
            validation_routine=self.validation_routine,
            metrics=self.metrics,
            data_generator=self.data_generator,
            schedulers=self.schedulers,
            plotter=self.plotter,
            save_dir=args.model_save_dir,
            title=args.title,
            report_train_every=10,
            plot_train_every=50
        )

        runner.run(number_of_epochs=args.epochs)

    def eval(self, args, print_every=1000):
        self.metrics.drop_state()
        self.model.eval()
        self.metrics.eval()
        it = 0
        for iter_data in self.data_generator.get_eval_generator():
            metrics_values = self.validation_routine.run(
                iter_num=it,
                iter_data=iter_data
            )
            self.metrics.report(metrics_values)

            if it % print_every == 0:
                self.metrics.get_current_value(should_print=True)
            it += 1

        self.metrics.decrease_hits(self.data_generator.data_reader.eval_tails)  # TODO: maybe cleaner?
        self.metrics.get_current_value(should_print=True)

    def create_terminal_embeddings(self, args):
        return create_terminal_embeddings(args)

    def create_non_terminal_embeddings(self, args):
        return None

    def create_optimizers(self, args):
        return get_optimizers(args, self.model)

    def create_schedulers(self, args):
        return [get_scheduler_args(args, opt) for opt in self.optimizers]

    def create_criterion(self, args):
        return nn.NLLLoss()

    def create_data_generator(self, args):
        return create_data_generator(args)

    def load_model(self, args):
        if args.saved_model is not None:
            if args.cuda:
                load_if_saved(self.model, args.saved_model)
            else:
                load_cuda_on_cpu(self.model, args.saved_model)
