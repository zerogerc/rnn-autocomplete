from torch.autograd import Variable

from zerogercrnn.lib.train.routines import NetworkRoutine


class ASTRoutine(NetworkRoutine):
    def __init__(self, network, criterion, optimizer=None):
        super(ASTRoutine, self).__init__(network)
        self.criterion = criterion
        self.optimizer = optimizer

    def run(self, iter_num, n_input, n_target):
        """Input and target here are pair of tensors (N, T)"""
        non_terminal_input = Variable(n_input[0].unsqueeze(2))
        terminal_input = Variable(n_target[0].unsqueeze(2))

        self.network.zero_grad()
        n_target = (Variable(n_target[0]), Variable(n_target[1]))
        n_output = self.network(non_terminal_input, terminal_input)

        loss = self.criterion(n_output, n_target)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()

        return loss.data[0]
