import torch
from torch.autograd import Variable

from zerogercrnn.experiments.js.ast_level.model.n2n_sum_attention_sequential import NTSumlAttentionModelSequential
from zerogercrnn.experiments.js.ast_level.model.utils import repackage_hidden, forget_hidden_partly_lstm_cell
from zerogercrnn.lib.train.routines import NetworkRoutine
from zerogercrnn.lib.utils.time import logger


def run_model(cuda, batch_size, model, iter_data, hidden):
    logger.reset_time()

    (n_input, n_target), forget_vector = iter_data
    assert forget_vector.size()[0] == batch_size

    non_terminal_input = Variable(n_input[0].unsqueeze(2))
    non_terminal_target = Variable(n_target[0])

    if cuda:
        non_terminal_input = non_terminal_input.cuda()
        non_terminal_target = non_terminal_target.cuda()

    logger.log_time_ms('TIME FOR GET DATA')

    if hidden is None:
        hidden = model.init_hidden(batch_size=batch_size, cuda=cuda)

    hidden = repackage_hidden(hidden)
    hidden = forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector)
    model.forget_context_partly(forget_vector=forget_vector)
    model.zero_grad()

    n_output = []
    for t in range(n_target[0].size()[0]):
        reinit_dropout = (t == 0)

        if t % 5 == 0:
            hidden = repackage_hidden(hidden)

        cur_output, hidden = model(
            non_terminal_input[t],
            hidden,
            forget_vector=forget_vector,
            reinit_dropout=reinit_dropout
        )

        n_output.append(cur_output.unsqueeze(0))

    n_output = torch.cat(n_output, dim=0)
    logger.log_time_ms('TIME FOR NETWORK')

    return n_output, non_terminal_target, hidden


class N2NSequential(NetworkRoutine):
    def __init__(self, model: NTSumlAttentionModelSequential, batch_size, seq_len, criterion, optimizers=None,
                 cuda=True):
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

        loss = self.criterion(n_output.permute(1, 2, 0), n_target.transpose(1, 0))
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
