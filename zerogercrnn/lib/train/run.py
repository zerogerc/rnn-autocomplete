import os
from tqdm import tqdm

from zerogercrnn.lib.train.routines import BaseRoutine
from zerogercrnn.lib.visualization.plotter import MatplotlibPlotter, VisdomPlotter
from zerogercrnn.lib.utils.state import save_model


class TrainEpochRunner:
    def __init__(self, network, loss_calc, optimizer, batcher, scheduler=None, plotter='matplotlib', save_dir=None):
        """Create train runner.
        
        :param network: network to train.
        :param loss_calc: function that computes loss for the given output and target.
        :param optimizer: optimizer to perform training. Should be bound to criterion.
        :param batcher: batcher that contains train and validation data. Expects train data to be stored by key *train*
            and validation data to be stored by key *validation*.
        :param scheduler: scheduler for learning rate. If None learning rate will be constant
        :param plotter: visualization tool. Either 'matplotlib' or 'visdom'.
        :param save_dir: if specified model will be saved in this directory after each epoch with name "model_epoch_X".
        """
        self.network = network
        self.optimizer = optimizer
        self.batcher = batcher
        self.scheduler = scheduler
        self.save_dir = save_dir

        if plotter == 'matplotlib':
            self.plotter = MatplotlibPlotter(title='linux')
        elif plotter == 'visdom':
            self.plotter = VisdomPlotter(title='linux', plots=['train', 'validation'])
        else:
            raise Exception('Unknown plotter')

        self.train_routine = BaseRoutine(
            network=network,
            criterion=loss_calc,
            optimizer=optimizer
        )
        self.validation_routine = BaseRoutine(
            network=network,
            criterion=loss_calc
        )

    def run(self, number_of_epochs, batch_size):
        it = 0
        self.validate(epoch=-1, iter_num=it, batch_size=batch_size)  # first validation for plot.

        train_generator = self.batcher.data_map['train']
        try:
            for epoch in tqdm(range(number_of_epochs)):
                if self.scheduler is not None:
                    self.scheduler.step()
                train_data = train_generator.get_batched_epoch(batch_size)
                # print('Expected number of iterations for epoch: {}'.format(train_generator.size // batch_size))

                for input_tensor, target_tensor in train_data:
                    loss = self.train_routine.run(
                        iter_num=it,
                        input_tensor=input_tensor,
                        target_tensor=target_tensor
                    )

                    self.plotter.on_new_point(
                        label='train',
                        x=it,
                        y=loss
                    )

                    it += 1

                # validate at the end of epoch
                self.validate(epoch=epoch, iter_num=it, batch_size=batch_size)

                if self.save_dir is not None:
                    save_model(
                        model=self.network,
                        path=os.path.join(self.save_dir, 'model_epoch_{}'.format(epoch))
                    )
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        finally:
            # plot graphs of validation and train losses
            self.plotter.on_finish()

    def validate(self, epoch, iter_num, batch_size):
        """Perform validation and calculate loss as an average of the whole validation dataset."""
        validation_data = self.batcher.data_map['validation'].get_batched_epoch(batch_size)

        total_loss = 0
        total_count = 0

        for input_tensor, target_tensor in validation_data:
            current_loss = self.validation_routine.run(
                iter_num=iter_num,
                input_tensor=input_tensor,
                target_tensor=target_tensor
            )

            total_loss += current_loss
            total_count += 1

        self.plotter.on_new_point(
            label='validation',
            x=iter_num,
            y=total_loss / total_count
        )

        print('Epoch: {}, Average loss: {}'.format(epoch, total_loss / total_count))
