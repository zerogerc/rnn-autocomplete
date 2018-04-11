import torch
from torch.autograd import Variable

from zerogercrnn.experiments.js.ast_level.model.n2n_attention import NTTailAttentionModel
from zerogercrnn.experiments.js.ast_level.model.n2n_sum_attention_sequential import NTSumlAttentionModelSequential
from zerogercrnn.experiments.js.ast_level.model.utils import forget_hidden_partly, repackage_hidden
from zerogercrnn.lib.train.routines import NetworkRoutine
from zerogercrnn.lib.utils.time import logger


def run_model(cuda, batch_size, model, iter_data, hidden):
    logger.reset_time()

    (n_input, n_target), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    non_terminal_input = Variable(n_input[0].unsqueeze(2))
    non_terminal_target = Variable(n_target[0][0])

    if cuda:
        non_terminal_input = non_terminal_input.cuda()
        non_terminal_target = non_terminal_target.cuda()

    logger.log_time_ms('TIME FOR GET DATA')

    model.zero_grad()

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size, cuda=cuda)

    # TODO: Maybe not reinit dropout every time?
    n_output, hidden = model(non_terminal_input, hidden, forget_vector=forget_vector, reinit_dropout=True)

    logger.log_time_ms('TIME FOR NETWORK')

    return n_output, non_terminal_target, hidden


class N2NSequential(NetworkRoutine):
    def __init__(self, model: NTSumlAttentionModelSequential, batch_size, seq_len, criterion, optimizers=None, cuda=True):
        super(N2NSequential, self).__init__(model)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.criterion = criterion
        self.optimizers = optimizers
        self.cuda = cuda and torch.cuda.is_available()

        self.clip_params = self.network.lstm_cell.parameters()
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
        if (self.optimizers is not None) and (iter_num % 50 == 0):
            # Backward pass
            loss.backward()

            # Clip recurrent core of network
            torch.nn.utils.clip_grad_norm(self.clip_params, self.clip_value)

            # Optimizer step
            for optimizer in self.optimizers:
                optimizer.step()

            self.hidden = repackage_hidden(hidden)

        logger.log_time_ms('TIME FOR CRITERION, BACKWARD, OPTIMIZER')
        # Return loss value
        return 0.1
