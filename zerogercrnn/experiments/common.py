from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.lib.core import BaseModule
from zerogercrnn.lib.data import BatchedDataGenerator
from zerogercrnn.lib.file import load_if_saved, load_cuda_on_cpu
from zerogercrnn.lib.metrics import Metrics
from zerogercrnn.lib.run import TrainEpochRunner, NetworkRoutine
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
        self.train_metrics = self.create_train_metrics(args)
        self.eval_metrics = self.create_eval_metrics(args)
        self.plotter = 'tensorboard'

    @abstractmethod
    def create_data_generator(self, args) -> BatchedDataGenerator:
        pass

    @abstractmethod
    def create_model(self, args) -> BaseModule:
        pass

    @abstractmethod
    def create_criterion(self, args) -> nn.Module:
        pass

    @abstractmethod
    def create_train_routine(self, args) -> NetworkRoutine:
        pass

    @abstractmethod
    def create_validation_routine(self, args) -> NetworkRoutine:
        pass

    @abstractmethod
    def create_train_metrics(self, args) -> Metrics:
        pass

    @abstractmethod
    def create_eval_metrics(self, args) -> Metrics:
        return self.create_train_metrics(args)  # Good enough if you don't want to eval now

    def train(self, args):
        runner = TrainEpochRunner(
            network=self.model,
            train_routine=self.train_routine,
            validation_routine=self.validation_routine,
            metrics=self.train_metrics,
            data_generator=self.data_generator,
            schedulers=self.schedulers,
            plotter=self.plotter,
            save_dir=args.model_save_dir,
            title=args.title,
            report_train_every=10,
            plot_train_every=50,
            save_model_every=args.save_model_every
        )

        runner.run(number_of_epochs=args.epochs)

    def eval(self, args):
        self.model.eval()
        self.eval_metrics.eval()
        self.eval_metrics.drop_state()
        it = 0
        hook_metrics = self.register_eval_hooks()

        with torch.no_grad():
            for iter_data in self.data_generator.get_eval_generator():
                metrics_values = self.validation_routine.run(
                    iter_num=it,
                    iter_data=iter_data
                )
                self.eval_metrics.report(metrics_values)
                it += 1

        self.eval_metrics.decrease_hits(self.data_generator.data_reader.eval_tails)
        self.eval_metrics.get_current_value(should_print=True)

        for m in hook_metrics:
            m.get_current_value(should_print=True)

    def register_eval_hooks(self):
        return []

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
