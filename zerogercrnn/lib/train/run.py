from tqdm import tqdm

from zerogercrnn.lib.train.routines import RNNTrainRoutine, RNNValidationRoutine
from zerogercrnn.lib.visualization.plotter import MatplotlibPlotter, VisdomPlotter


class TrainEpochRunner:
    def __init__(self, network, loss_calc, optimizer, batcher, scheduler=None):
        """Create train runner.
        
        :param network: network to train.
        :param loss_calc: function that computes loss for the given output and target.
        :param optimizer: optimizer to perform training. Should be bound to criterion.
        :param batcher: batcher that contains train and validation data. Expects train data to be stored by key *train*
            and validation data to be stored by key *validation*.
        """
        self.network = network
        self.optimizer = optimizer
        self.batcher = batcher
        self.scheduler = scheduler

        self.plotter = MatplotlibPlotter(title='linux')  # Visdom: plots=['train', 'validation']
        self.train_routine = RNNTrainRoutine(
            label='train',
            network=network,
            plotter=self.plotter,
            loss_calc=loss_calc,
            optimizer=optimizer
        )
        self.validation_routine = RNNValidationRoutine(
            label='validation',
            network=network,
            plotter=self.plotter,
            loss_calc=loss_calc
        )

    def run(self, number_of_epochs, batch_size):
        validation_random_batches = self.batcher.data_map['validation'].get_batched_random(batch_size)

        it = 0
        self.validate(it, validation_random_batches)  # first validation for plot.

        train_generator = self.batcher.data_map['train']
        try:
            for epoch in tqdm(range(number_of_epochs)):
                if self.scheduler is not None:
                    self.scheduler.step()
                train_data = train_generator.get_batched_epoch(batch_size)
                # print('Expected number of iterations for epoch: {}'.format(train_generator.size // batch_size))

                for input_tensor, target_tensor in train_data:
                    self.train_routine.run(iter_num=it, input_tensor=input_tensor, target_tensor=target_tensor)
                    it += 1

                # validate at the end of epoch
                self.validate(it, validation_random_batches)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        finally:
            # plot graphs of validation and train losses
            self.plotter.on_finish()

    def validate(self, iter_num, validation_random_batches):
        val_input, val_target = next(validation_random_batches)
        self.validation_routine.run(iter_num=iter_num, input_tensor=val_input, target_tensor=val_target)
