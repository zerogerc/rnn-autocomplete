from torch.autograd import Variable


class NetworkRoutine:
    """Base class for running single iteration of RNN. Enable to train or validate networks."""

    def __init__(self, network):
        self.network = network

    def run(self, iter_num, n_input, n_target):
        """ Run routine and return value of loss function.
        
        :param iter_num: number of iteration
        :param n_input: input corresponding to network.
        :param n_target: output corresponding to criterion and network.
        """
        pass


class BaseRoutine(NetworkRoutine):
    """Base routine for training/validation of networks where loss function is cal."""

    def __init__(self, network, criterion, optimizer=None):
        """
        :param network: nn.Module.
        :param criterion: function that computes loss for the given output and target.
        :param optimizer: is specified routine will perform backward pass and optimizer step.
        """
        super(BaseRoutine, self).__init__(network)
        self.criterion = criterion
        self.optimizer = optimizer

    def run(self, iter_num, n_input, n_target):
        """Run network and return float value of loss.
        
        :param iter_num: number of iteration (used to plot loss).
        :param n_input: tensor of size [seq_len, batch_size, input_size].
        :param n_target: tensor of size [seq_len, batch_size, target_size]. 
        :return: float value of loss computed with specified loss function.
        """
        n_input = Variable(n_input)
        n_target = Variable(n_target)

        self.network.zero_grad()
        output_tensor = self.network(n_input)

        loss = self.criterion(output_tensor, n_target)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()

        return loss.data[0]
