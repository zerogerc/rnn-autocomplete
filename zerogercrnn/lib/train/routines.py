from torch.autograd import Variable

class NetworkRoutine:
    """Base class for running single iteration of RNN. Enable to train or validate networks."""

    def __init__(self, network):
        self.network = network

    def run(self, iter_num, input_tensor, target_tensor):
        """ Run routine.
        
        :param iter_num: number of iteration
        :param input_tensor: tensor of the size corresponding to network.
        :param target_tensor: tensor of the size corresponding to criterion and network.
        """
        pass


class PlottedNetworkRoutine(NetworkRoutine):
    """Network that will run on iteration and store loss to the plot."""

    def __init__(self, network):
        super(PlottedNetworkRoutine, self).__init__(network)
        self.plot = PlotData()  # place to store loss on particular iteration.

    def run(self, iter_num, input_tensor, target_tensor):
        """Run routine and store loss on the plot.
        
        :return: float value of loss.
        """
        input_tensor = Variable(input_tensor)
        target_tensor = Variable(target_tensor)
        loss = self.__run_and_calc_loss__(iter_num, input_tensor, target_tensor)
        self.plot.add(iter_num, loss)
        return loss

    def __run_and_calc_loss__(self, iter_num, input_tensor, target_tensor):
        """Run routine and return float value of loss on input_tensor."""
        return 0


class RNNRoutine(PlottedNetworkRoutine):
    """Base routine for training/validation of RNN networks."""

    def __init__(self, network, loss_calc):
        """
        :param loss_calc: function that computes loss for the given output and target.
        """
        super(RNNRoutine, self).__init__(network)
        self.loss_calc = loss_calc

    def __run_and_calc_loss__(self, iter_num, input_tensor, target_tensor):
        """Run network and plot loss.
        
        :param iter_num: number of iteration (used to plot loss).
        :param input_tensor: tensor of size [seq_len, batch_size, input_size].
        :param target_tensor: tensor of size [seq_len, batch_size, target_size]. 
        :return: float value of loss computed with specified loss function.
        """
        self.network.zero_grad()
        output_tensor = self.network(input_tensor)

        loss = self.loss_calc(output_tensor, target_tensor)

        return self.__process_loss__(loss)

    def __process_loss__(self, loss):
        """Process loss and do backward path if necessary."""
        return loss.data[0]


class RNNTrainRoutine(RNNRoutine):
    """Routine for training RNN networks."""

    def __init__(self, network, loss_calc, optimizer):
        """Create training routine.
        
        :param network: network to train.
        :param loss_calc: function that computes loss for the given output and target.
        :param optimizer: optimizer to perform training. Should be bound to criterion.
        """
        super(RNNTrainRoutine, self).__init__(network, loss_calc)
        self.optimizer = optimizer

    def __process_loss__(self, loss):
        loss.backward()
        self.optimizer.step()
        return loss.data[0]


class RNNValidationRoutine(RNNRoutine):
    """Routine for validation RNN networks. This routine don't run backward path."""

    def __init__(self, network, loss_calc):
        """
        :param network: network to train.
        :param loss_calc: function that computes loss for the given output and target.
        """
        super(RNNValidationRoutine, self).__init__(network, loss_calc)

    def __process_loss__(self, loss):
        print('validation loss: {}'.format(loss))
        return loss.data[0]


class PlotData:
    def __init__(self):
        self.x = []
        self.y = []

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)
