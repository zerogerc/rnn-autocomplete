from abc import abstractmethod

from torch import nn as nn

from zerogercrnn.experiments.ast_level.data import ASTDataReader, ASTDataGenerator

from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.experiments.common import get_optimizers, get_scheduler_args
from zerogercrnn.lib.run import TrainEpochRunner
from zerogercrnn.lib.utils.state import load_if_saved, load_cuda_on_cpu


def create_terminal_embeddings(args):
    return Embeddings(
        embeddings_size=args.terminal_embedding_dim,
        vector_file=args.terminal_embeddings_file,
        squeeze=True
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
        self.terminal_embeddings = create_terminal_embeddings(args)

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

    def run(self, args):
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
            plot_train_every=50
        )

        runner.run(number_of_epochs=args.epochs)

    def create_terminals_embeddings(self, args):
        return create_terminal_embeddings(args)

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
