from torch.autograd import Variable


class NetworkRoutine:
    """Base class for running single iteration of RNN. Enable to train or validate networks."""

    def __init__(self, network):
        self.network = network

    def run(self, iter_num, input_tensor, target_tensor):
        """ Run routine and return value of loss function.
        
        :param iter_num: number of iteration
        :param input_tensor: tensor of the size corresponding to network.
        :param target_tensor: tensor of the size corresponding to criterion and network.
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

    def run(self, iter_num, input_tensor, target_tensor):
        """Run network and return float value of loss.
        
        :param iter_num: number of iteration (used to plot loss).
        :param input_tensor: tensor of size [seq_len, batch_size, input_size].
        :param target_tensor: tensor of size [seq_len, batch_size, target_size]. 
        :return: float value of loss computed with specified loss function.
        """
        input_tensor = Variable(input_tensor)
        target_tensor = Variable(target_tensor)

        self.network.zero_grad()
        output_tensor = self.network(input_tensor)

        loss = self.criterion(output_tensor, target_tensor)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()

        return loss.data[0]
