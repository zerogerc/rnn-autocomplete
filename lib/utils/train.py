import time

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from lib.data.batcher import Batcher
from lib.utils.time import time_since


def default_loss_calculator(output_tensor, target_tensor, criterion, seq_len):
    """
    Default way to compute loss for rnn.
    It's the mean of losses on all timestamps.
    """
    loss = 0
    for i in range(seq_len):
        loss += criterion(output_tensor[i], target_tensor[i])
    return loss / seq_len


class RNNRunner:
    """
    Runner for typical RNN networks that accept input of size [seq_len, data_len, input_len] 
    and returns target of size [seq_len, data_len, target_len].
    
    seq_len is a parameter of constructor because it needed to compute loss
    
    After training will print to plots: validation loss, train loss
    """

    def __init__(
            self,
            network: nn.Module,
            optimizer: optim.Optimizer,
            criterion,
            batcher: Batcher,
            seq_len: int,
            loss_calculator=default_loss_calculator
    ):
        """
        Create RNNRunner. All parameters a pretty self explaining except loss_calculator.
        
        :param loss_calculator: function that computes loss on the output of rnn.
            See **default_loss_calculator** for example
        """
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.batcher = batcher
        self.seq_len = seq_len
        self.loss_calculator = loss_calculator

    def run_train(self, batch_size: int, n_iters: int, validation_every: int, print_every: int = 1000):
        """
        Run train with parameters passed in constructor.
        
        :param batch_size: size of chunks to split input during training
        :param n_iters: number of iterations to run
        :param validation_every: how often to run train on validation dataset
        :param print_every: how often to print current loss (loss on train) to the console
        """
        # train plot: x - iterations, y - loss
        train_losses_x = []
        train_losses_y = []

        # validation plot: x - iterations, y - loss
        validation_losses_x = []
        validation_losses_y = []

        start = time.time()

        # get train and validation data generators
        data_map = self.batcher.data_map
        train_data = data_map['train'].get_batched(batch_size)
        validation_data = data_map['validation'].get_batched(batch_size)
        test_data = data_map['test'].get_batched(batch_size)

        # run first time to initialize train and validation losses (without weight update)
        input_tensor, target_tensor = next(train_data)
        self.__run_iter__(0, input_tensor, target_tensor, train_losses_x, train_losses_y, do_backward=False)

        input_tensor, target_tensor = next(validation_data)
        self.__run_iter__(0, input_tensor, target_tensor, validation_losses_x, validation_losses_y, do_backward=False)

        # run all iters with weights update
        for iter_num in range(1, n_iters + 1):
            if iter_num % validation_every == 0:  # should run on validation set also
                input_tensor, target_tensor = next(validation_data)
                self.__run_iter__(iter_num, input_tensor, target_tensor, validation_losses_x, validation_losses_y)
            else:
                input_tensor, target_tensor = next(train_data)
                self.__run_iter__(iter_num, input_tensor, target_tensor, train_losses_x, train_losses_y)

            if iter_num % print_every == 0:
                print('%s (%d %d%%) %.4f' % (
                    time_since(start), iter_num, iter_num / n_iters * 100, validation_losses_y[-1])
                )

        # run on test
        input_tensor, target_tensor = next(test_data)
        test_loss = []
        self.__run_iter__(0, input_tensor, target_tensor, [], test_loss, do_backward=False)
        print('Loss on test data: {}'.format(test_loss[-1]))

        # plot graphs of validation and train losses
        plt.plot(train_losses_x, train_losses_y, label='Train')
        plt.plot(validation_losses_x, validation_losses_y, label='Validation')
        plt.legend()
        plt.show()

    def __run_iter__(self, iter_num, input_tensor, target_tensor, losses_x, losses_y, do_backward=True):
        """ Run single iteration. Append result to losses_x and losses_y. """
        self.network.zero_grad()
        output_tensor = self.network(input_tensor)

        loss = self.loss_calculator(output_tensor, target_tensor, self.criterion, self.seq_len)

        if do_backward:
            loss.backward()
            self.optimizer.step()

        losses_x.append(iter_num)
        losses_y.append(loss.data[0])
