import os
from abc import abstractmethod

from tqdm import tqdm

from zerogercrnn.lib.metrics import Metrics

# hack for tqdm
tqdm.monitor_interval = 0

import torch.nn as nn

from zerogercrnn.lib.visualization.plotter import TensorboardPlotter, \
    TensorboardPlotterCombined
from zerogercrnn.lib.file import save_model
from zerogercrnn.lib.data.general import DataGenerator

LOG_EVERY = 1000


class NetworkRoutine:
    """Base class for running single iteration of RNN. Enable to train or validate networks."""

    def __init__(self, network):
        self.network = network

    @abstractmethod
    def run(self, iter_num, iter_data):
        """ Run routine and return value for plotting.

        :param iter_num: number of iteration
        :param iter_data: data for this iteration
        """
        pass


def save_current_model(model, dir, name):
    if dir is not None:
        print('Saving model: {}'.format(name))
        save_model(
            model=model,
            path=os.path.join(dir, name)
        )
        print('Saved!')


class TrainEpochRunner:
    def __init__(
            self,
            network: nn.Module,
            train_routine: NetworkRoutine,
            validation_routine: NetworkRoutine,
            metrics: Metrics,
            data_generator: DataGenerator,
            schedulers=None,
            plotter='tensorboard',
            save_dir=None,
            title=None,
            plot_train_every=1,
            save_iter_model_every=None
    ):
        """Create train runner.
        
        :param network: network to train.
        :param train_routine: routine that will run on each train input.
        :param validation_routine: routine that will run after each epoch of training on each validation input.
        :param metrics: metrics to plot. Should correspond to routine
        :param data_generator: generator of data for training and validation
        :param schedulers: schedulers for learning rate. If None learning rate will be constant
        :param plotter: visualization tool. Either 'matplotlib' or 'visdom'.
        :param save_dir: if specified model will be saved in this directory after each epoch with name "model_epoch_X".
        :param title: used for visualization
        """
        self.network = network
        self.train_routine = train_routine
        self.validation_routine = validation_routine
        self.metrics = metrics
        self.data_generator = data_generator
        self.schedulers = schedulers
        self.save_dir = save_dir
        self.plot_train_every = plot_train_every
        self.save_iter_model_every = save_iter_model_every

        self.epoch = None  # current epoch
        self.it = None  # current iteration

        if plotter == 'tensorboard':
            self.plotter = TensorboardPlotter(title=title)
        elif plotter == 'tensorboard_combined':
            self.plotter = TensorboardPlotterCombined(title=title)
        else:
            raise Exception('Unknown plotter')

    def run(self, number_of_epochs):
        self.epoch = -1
        self.it = 0
        self._validate()  # first validation for plot.

        try:
            while self.epoch < number_of_epochs:
                self.epoch += 1
                if self.schedulers is not None:
                    for scheduler in self.schedulers:
                        scheduler.step()

                self._run_for_epoch()
                self._validate()

                save_current_model(self.network, self.save_dir, name='model_epoch_{}'.format(self.epoch))
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        finally:
            # plot graphs of validation and train losses
            self.plotter.on_finish()

    def _run_for_epoch(self):
        self.network.train()
        train_data = self.data_generator.get_train_generator()
        # print('Expected number of iterations for epoch: {}'.format(train_generator.size // batch_size))

        for iter_data in train_data:
            if self.it % LOG_EVERY == 0:
                print('Training... Epoch: {}, Iters: {}'.format(self.epoch, self.it))

            if (self.save_iter_model_every is not None) and (self.it % self.save_iter_model_every == 0):
                save_current_model(self.network, self.save_dir, name='model_iter_{}'.format(self.it))

            metrics_values = self.train_routine.run(
                iter_num=self.it,
                iter_data=iter_data
            )

            if self.it % self.plot_train_every == 0:
                self.metrics.drop_state()
                self.metrics.report(metrics_values)
                self.plotter.on_new_point(
                    label='train',
                    x=self.it,
                    y=self.metrics.get_current_value(should_print=False)
                )

            self.it += 1

    def _validate(self):
        """Perform validation and calculate loss as an average of the whole validation dataset."""
        validation_data = self.data_generator.get_validation_generator()

        self.metrics.drop_state()
        self.network.eval()

        validation_it = 0
        for iter_data in validation_data:
            if validation_it % LOG_EVERY == 0:
                print('Validating... Epoch: {} Iters: {}'.format(self.epoch, validation_it))

            metrics_values = self.validation_routine.run(
                iter_num=self.it,
                iter_data=iter_data
            )

            self.metrics.report(metrics_values)

            validation_it += 1

        self.plotter.on_new_point(
            label='validation',
            x=self.it,
            y=self.metrics.get_current_value(should_print=False)
        )

        print('Validation done. Epoch: {}'.format(self.epoch))
        self.metrics.get_current_value(should_print=True)
