import os

from tqdm import tqdm

# hack for tqdm
tqdm.monitor_interval = 0

from torch.autograd import Variable
import torch.nn as nn

from zerogercrnn.lib.visualization.plotter import MatplotlibPlotter, VisdomPlotter, TensorboardPlotter
from zerogercrnn.lib.utils.state import save_model
from zerogercrnn.lib.data.general import DataGenerator
from zerogercrnn.lib.train.routines import NetworkRoutine

LOG_EVERY = 1000


class TrainEpochRunner:
    def __init__(
            self,
            network: nn.Module,
            train_routine: NetworkRoutine,
            validation_routine: NetworkRoutine,
            data_generator: DataGenerator,
            schedulers=None,
            plotter='matplotlib',
            save_dir=None,
            title='TrainRunner',
            plot_train_every=1
    ):
        """Create train runner.
        
        :param network: network to train.
        :param train_routine: routine that will run on each train input.
        :param validation_routine: routine that will run after each epoch of training on each validation input.
        :param data_generator: generator of data for training and validation
        :param schedulers: schedulers for learning rate. If None learning rate will be constant
        :param plotter: visualization tool. Either 'matplotlib' or 'visdom'.
        :param save_dir: if specified model will be saved in this directory after each epoch with name "model_epoch_X".
        :param title: used for visualization
        """
        self.network = network
        self.train_routine = train_routine
        self.validation_routine = validation_routine
        self.data_generator = data_generator
        self.schedulers = schedulers
        self.save_dir = save_dir
        self.skip_train_points = plot_train_every

        if plotter == 'matplotlib':
            self.plotter = MatplotlibPlotter(title=title)
        elif plotter == 'visdom':
            self.plotter = VisdomPlotter(title=title, plots=['train', 'validation'])
        elif plotter == 'tensorboard':
            self.plotter = TensorboardPlotter(title=title)
        else:
            raise Exception('Unknown plotter')

    def run(self, number_of_epochs):
        it = 0
        # self.validate(epoch=-1, iter_num=it)  # first validation for plot.

        try:
            for epoch in range(number_of_epochs):
                if self.schedulers is not None:
                    for scheduler in self.schedulers:
                        scheduler.step()
                train_data = self.data_generator.get_train_generator()
                # print('Expected number of iterations for epoch: {}'.format(train_generator.size // batch_size))

                epoch_it = 0
                for iter_data in train_data:
                    if epoch_it % LOG_EVERY == 0:
                        print('Training... Epoch: {}, Iters: {}'.format(epoch, it))

                    loss = self.train_routine.run(
                        iter_num=it,
                        iter_data=iter_data
                    )

                    if epoch_it % self.skip_train_points == 0:
                        if isinstance(loss, Variable):
                            loss = loss.data[0]

                        self.plotter.on_new_point(
                            label='train',
                            x=it,
                            y=loss
                        )

                    epoch_it += 1
                    it += 1

                # validate at the end of epoch
                self.validate(epoch=epoch, iter_num=it)

                if self.save_dir is not None:
                    print('Saving model: {}'.format('model_epoch_{}'.format(epoch)))
                    save_model(
                        model=self.network,
                        path=os.path.join(self.save_dir, 'model_epoch_{}'.format(epoch))
                    )
                    print('Saved!')
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        finally:
            # plot graphs of validation and train losses
            self.plotter.on_finish()

    def validate(self, epoch, iter_num):
        """Perform validation and calculate loss as an average of the whole validation dataset."""
        validation_data = self.data_generator.get_validation_generator()

        total_loss = None
        total_count = 0

        for iter_data in validation_data:
            if total_count % LOG_EVERY == 0:
                print('Validating... Epoch: {} Iters: {}'.format(epoch, total_count))
            current_loss = self.validation_routine.run(
                iter_num=iter_num,
                iter_data=iter_data
            )

            if total_loss is None:
                total_loss = current_loss
            else:
                total_loss += current_loss

            total_count += 1

        if isinstance(total_loss, Variable):
            total_loss = total_loss.data[0]

        self.plotter.on_new_point(
            label='validation',
            x=iter_num,
            y=total_loss / total_count
        )

        print('Validation done. Epoch: {}, Average loss: {}'.format(epoch, total_loss / total_count))
