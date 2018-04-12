import torch
from torch.autograd import Variable

from zerogercrnn.experiments.ast_level.model.n2n_attention import NTTailAttentionModel
from zerogercrnn.experiments.ast_level.model.utils import forget_hidden_partly, repackage_hidden
from zerogercrnn.lib.train.routines import NetworkRoutine
from zerogercrnn.lib.utils.time import logger


def run_model(cuda, batch_size, model, iter_data, hidden):
    logger.reset_time()

    (n_input, n_target), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    non_terminal_input = Variable(n_input[0].unsqueeze(2))
    non_terminal_target = Variable(n_target[0][-1])

    if cuda:
        non_terminal_input = non_terminal_input.cuda()
        non_terminal_target = non_terminal_target.cuda()

    logger.log_time_ms('TIME FOR GET DATA')

    model.zero_grad()

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size, cuda=cuda)

    hidden = forget_hidden_partly(hidden, forget_vector)

    n_output, hidden = model(non_terminal_input, hidden)
    hidden = repackage_hidden(hidden)

    logger.log_time_ms('TIME FOR NETWORK')

    return n_output, non_terminal_target, hidden


class NTTailAttentionASTRoutine(NetworkRoutine):
    def __init__(self, model: NTTailAttentionModel, batch_size, criterion, optimizers=None, cuda=True):
        super(NTTailAttentionASTRoutine, self).__init__(model)
        assert self.network.recurrent is not None

        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizers = optimizers
        self.cuda = cuda and torch.cuda.is_available()

        self.clip_params = self.network.recurrent.parameters()
        self.clip_value = 5.

        self.hidden = None

    def run(self, iter_num, iter_data):
        n_output, n_target, hidden = run_model(
            cuda=self.cuda,
            batch_size=self.batch_size,
            model=self.network,
            iter_data=iter_data,
            hidden=self.hidden
        )
        self.hidden = hidden

        loss = self.criterion(n_output, n_target)
        if self.optimizers is not None:
            # Backward pass
            loss.backward()

            # Clip recurrent core of network
            torch.nn.utils.clip_grad_norm(self.clip_params, self.clip_value)

            # Optimizer step
            for optimizer in self.optimizers:
                optimizer.step()

        logger.log_time_ms('TIME FOR CRITERION, BACKWARD, OPTIMIZER')
        # Return loss value
        return loss.data[0]
