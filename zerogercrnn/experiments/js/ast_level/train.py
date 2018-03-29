import time
import torch
from torch.autograd import Variable

from zerogercrnn.lib.train.routines import NetworkRoutine


def print_interval_and_update(label, ct):
    print("{}: {}".format(label, (1000 * (time.clock() - ct))))
    return time.clock()


class ASTRoutine(NetworkRoutine):
    def __init__(self, network, criterion, optimizers=None, cuda=True):
        super(ASTRoutine, self).__init__(network)
        self.criterion = criterion
        self.optimizers = optimizers
        self.cuda = cuda and torch.cuda.is_available()

    def run(self, iter_num, n_input, n_target):
        """Input and target here are pair of tensors (N, T)"""

        ct = time.clock()
        non_terminal_input = Variable(n_input[0].unsqueeze(2))
        terminal_input = Variable(n_target[0].unsqueeze(2))

        non_terminal_target = Variable(n_target[0])
        terminal_target = Variable(n_target[1])

        if self.cuda:
            non_terminal_input = non_terminal_input.cuda()
            terminal_input = terminal_input.cuda()
            non_terminal_target = non_terminal_target.cuda()
            terminal_target = terminal_target.cuda()

        ct = print_interval_and_update("TIME GET DATA", ct)

        self.network.zero_grad()
        ct = print_interval_and_update("ZERO_GRAD", ct)
        n_target = (non_terminal_target, terminal_target)
        ct = print_interval_and_update("CORTAGE", ct)
        n_output = self.network(non_terminal_input, terminal_input)

        ct = print_interval_and_update("TIME NETWORK", ct)

        loss = self.criterion(n_output, n_target)

        ct = print_interval_and_update("TIME CRITERION", ct)

        if self.optimizers is not None:
            loss.backward()
            ct = print_interval_and_update("TIME BACKWARD", ct)
            for optimizer in self.optimizers:
                optimizer.step()
            ct = print_interval_and_update("TIME STEP", ct)

        return 0
