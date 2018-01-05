import math
import time

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from lib.data.batcher import Batcher


class PlotData:
    def __init__(self):
        self.x = []
        self.y = []

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)


class RNNRunner:
    """Runner for typical RNN networks that accept input of size [seq_len, data_len, input_len] 
    and returns target of size [seq_len, data_len, target_len].
    
    seq_len is a parameter of constructor because it needed to compute loss.
    
    After training will print two plots: validation loss, train loss
    """

    def __init__(
            self,
            network: nn.Module,
            optimizer: optim.Optimizer,
            criterion,
            batcher: Batcher,
            seq_len: int
    ):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.batcher = batcher
        self.seq_len = seq_len

    def run_epoch(self, validation_every, train_epoch, validation_random, test_random=None):
        train_plot = PlotData()
        validation_plot = PlotData()

        cur_iter = 0
        start = time.time()

        for input_tensor, target_tensor in train_epoch:
            if cur_iter % validation_every == 0:
                val_input, val_target = next(validation_random)
                self.__run_iter__(cur_iter, val_input, val_target, validation_plot,
                                  do_backward=False)
                print('Validation:\n time: %s iter: %d loss: %.4f' % (
                    time_since(start), cur_iter, validation_plot.y[-1])
                      )

            self.__run_iter__(cur_iter, input_tensor, target_tensor, train_plot)
            cur_iter += 1

        if test_random is not None:
            input_tensor, target_tensor = next(test_random)
            test_data = PlotData()
            self.__run_iter__(0, input_tensor, target_tensor, test_data, do_backward=False)
            print('Loss on test data: {}'.format(test_data.y[-1]))

        # plot graphs of validation and train losses
        plt.plot(train_plot.x, train_plot.y, label='Train')
        plt.plot(validation_plot.x, validation_plot.y, label='Validation')
        plt.legend()
        plt.show()

    def run_train(self, batch_size: int, n_iters: int, validation_every: int, print_every: int = 1000):
        """Run train with parameters passed in constructor.
        
        :param batch_size: size of chunks to split input during training
        :param n_iters: number of iterations to run
        :param validation_every: how often to run train on validation dataset
        :param print_every: how often to print current loss (loss on train) to the console
        """
        train_plot = PlotData()
        validation_plot = PlotData()

        start = time.time()

        # get train and validation data generators
        data_map = self.batcher.data_map
        train_data = data_map['train'].get_batched_random(batch_size)
        validation_data = data_map['validation'].get_batched_random(batch_size)
        test_data = data_map['test'].get_batched_random(batch_size)

        # run first time to initialize train and validation losses (without weight update)
        input_tensor, target_tensor = next(train_data)
        self.__run_iter__(0, input_tensor, target_tensor, train_plot, do_backward=False)

        input_tensor, target_tensor = next(validation_data)
        self.__run_iter__(0, input_tensor, target_tensor, validation_plot, do_backward=False)

        # run all iters with weights update
        for iter_num in range(1, n_iters + 1):
            if iter_num % validation_every == 0:  # should run on validation set also
                input_tensor, target_tensor = next(validation_data)
                self.__run_iter__(iter_num, input_tensor, target_tensor, validation_plot)
            else:
                input_tensor, target_tensor = next(train_data)
                self.__run_iter__(iter_num, input_tensor, target_tensor, train_plot)

            if iter_num % print_every == 0:
                print('%s (%d %d%%) %.4f' % (
                    time_since(start), iter_num, iter_num / n_iters * 100, validation_plot.y[-1])
                      )

        # run on test
        input_tensor, target_tensor = next(test_data)
        test_loss = PlotData()
        self.__run_iter__(0, input_tensor, target_tensor, test_loss, do_backward=False)
        print('Loss on test data: {}'.format(test_loss.y[-1]))

        # plot graphs of validation and train losses
        plt.plot(train_plot.x, train_plot.y, label='Train')
        plt.plot(validation_plot.x, validation_plot.y, label='Validation')
        plt.legend()
        plt.show()

    def __run_iter__(self, iter_num, input_tensor, target_tensor, losses, do_backward=True):
        """ Run single iteration. Append result to losses_x and losses_y. """
        self.network.zero_grad()
        output_tensor = self.network(input_tensor)

        loss = self.__calc_loss__(output_tensor, target_tensor)

        if do_backward:
            loss.backward()
            self.optimizer.step()

        losses.add(iter_num, loss.data[0])

    def __calc_loss__(self, output_tensor, target_tensor):
        # flatten tensors
        sz_o = output_tensor.size()[-1]
        sz_t = target_tensor.size()[-1]

        return self.criterion(output_tensor.view(-1, sz_o), target_tensor.view(-1, sz_t))


def time_since(since):
    """Calculates time in seconds since given time."""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
