from abc import abstractmethod

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.lib.file import load_if_saved, load_cuda_on_cpu
from zerogercrnn.lib.run import TrainEpochRunner
from zerogercrnn.lib.utils import filter_requires_grad, get_best_device


# region CreateUtils

def get_optimizer(args, model):
    return optim.Adam(
        params=filter_requires_grad(model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )


def get_sparse_optimizer(args, model):
    return optim.SparseAdam(
        params=filter_requires_grad(model.sparse_parameters()),
        lr=args.learning_rate
    )


def get_optimizers(args, model):
    optimizers = []
    if len(list(filter_requires_grad(model.parameters()))) != 0:
        optimizers.append(get_optimizer(args, model))
    if len(list(filter_requires_grad(model.sparse_parameters()))) != 0:
        optimizers.append(get_sparse_optimizer(args, model))

    if len(optimizers) == 0:
        raise Exception('Model has no parameters!')

    return optimizers


def get_scheduler(args, optimizer):
    return MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(args.decay_after_epoch, args.epochs + 20)),
        gamma=args.decay_multiplier
    )


# endregion


class Main:
    def __init__(self, args):
        self.model = self.create_model(args).to(get_best_device())
        self.load_model(args)

        self.optimizers = self.create_optimizers(args)
        self.schedulers = self.create_schedulers(args)
        self.criterion = self.create_criterion(args)

        self.data_generator = self.create_data_generator(args)

        self.train_routine = self.create_train_routine(args)
        self.validation_routine = self.create_validation_routine(args)
        self.metrics = self.create_metrics(args)
        self.plotter = 'tensorboard'

    @abstractmethod
    def create_data_generator(self, args):
        pass

    @abstractmethod
    def create_model(self, args):
        pass

    @abstractmethod
    def create_criterion(self, args):
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
        with torch.no_grad():
            for iter_data in self.data_generator.get_eval_generator():
                metrics_values = self.validation_routine.run(
                    iter_num=it,
                    iter_data=iter_data
                )
                self.metrics.report(metrics_values)

                # if it % print_every == 0:
                #     self.metrics.get_current_value(should_print=True)
                it += 1

        self.metrics.get_current_value(should_print=True)
        self.metrics.decrease_hits(self.data_generator.data_reader.eval_tails)  # TODO: maybe cleaner?
        self.metrics.get_current_value(should_print=True)

    def create_optimizers(self, args):
        return get_optimizers(args, self.model)

    def create_schedulers(self, args):
        return [get_scheduler(args, opt) for opt in self.optimizers]

    def load_model(self, args):
        if args.saved_model is not None:
            if torch.cuda.is_available():
                load_if_saved(self.model, args.saved_model)
            else:
                load_cuda_on_cpu(self.model, args.saved_model)
