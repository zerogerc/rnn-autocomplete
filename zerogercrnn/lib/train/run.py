import os
from tqdm import tqdm

import torch
import gc

from torch.autograd import Variable
import torch.nn as nn

from zerogercrnn.lib.train.routines import BaseRoutine
from zerogercrnn.lib.visualization.plotter import MatplotlibPlotter, VisdomPlotter, TensorboardPlotter
from zerogercrnn.lib.utils.state import save_model
from zerogercrnn.lib.data.general import DataGenerator
from zerogercrnn.lib.train.routines import NetworkRoutine


class TrainEpochRunner:
    def __init__(
            self,
            network: nn.Module,
            train_routine: NetworkRoutine,
            validation_routine: NetworkRoutine,
            data_generator: DataGenerator,
            scheduler=None,
            plotter='matplotlib',
            save_dir=None,
            title='TrainRunner',
            skip_train_points=100
    ):
        """Create train runner.
        
        :param network: network to train.
        :param train_routine: routine that will run on each train input.
        :param validation_routine: routine that will run after each epoch of training on each validation input.
        :param data_generator: generator of data for training and validation
        :param scheduler: scheduler for learning rate. If None learning rate will be constant
        :param plotter: visualization tool. Either 'matplotlib' or 'visdom'.
        :param save_dir: if specified model will be saved in this directory after each epoch with name "model_epoch_X".
        :param title: used for visualization
        """
        self.network = network
        self.train_routine = train_routine
        self.validation_routine = validation_routine
        self.data_generator = data_generator
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.skip_train_points = skip_train_points

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
        self.validate(epoch=-1, iter_num=it)  # first validation for plot.

        try:
            for epoch in tqdm(range(number_of_epochs)):
                if self.scheduler is not None:
                    self.scheduler.step()
                train_data = self.data_generator.get_train_generator()
                # print('Expected number of iterations for epoch: {}'.format(train_generator.size // batch_size))

                train_point_id = 0
                for n_input, n_target in train_data:
                    loss = self.train_routine.run(
                        iter_num=it,
                        n_input=n_input,
                        n_target=n_target
                    )

                    # if train_point_id % self.skip_train_points == 0:
                    #     if isinstance(loss, Variable):
                    #         loss = loss.data[0]

                    # self.plotter.on_new_point(
                    #     label='train',
                    #     x=it,
                    #     y=loss
                    # )

                    count = 0
                    for obj in gc.get_objects():
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            count += 1

                    print("Tensors: {}".format(count))

                    it += 1

                # validate at the end of epoch
                self.validate(epoch=epoch, iter_num=it)

                if self.save_dir is not None:
                    save_model(
                        model=self.network,
                        path=os.path.join(self.save_dir, 'model_epoch_{}'.format(epoch))
                    )
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        finally:
            x = 0
            # plot graphs of validation and train losses
            # self.plotter.on_finish()

    def validate(self, epoch, iter_num):
        """Perform validation and calculate loss as an average of the whole validation dataset."""
        validation_data = self.data_generator.get_validation_generator()

        total_loss = None
        total_count = 0

        for input_tensor, target_tensor in validation_data:
            current_loss = self.validation_routine.run(
                iter_num=iter_num,
                n_input=input_tensor,
                n_target=target_tensor
            )

            if total_loss is None:
                total_loss = current_loss
            else:
                total_loss += current_loss

            total_count += 1

        if isinstance(total_loss, Variable):
            total_loss = total_loss.data[0]

        # self.plotter.on_new_point(
        #     label='validation',
        #     x=iter_num,
        #     y=total_loss / total_count
        # )

        print('Epoch: {}, Average loss: {}'.format(epoch, total_loss / total_count))
