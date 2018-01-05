import torch
import torch.nn as nn
from torch.autograd import Variable

from lib.module.RNN import RNN


class TrendsRNN(nn.Module):
    """Network with multi-layer RNN."""

    def __init__(self, input_size, hidden_size, output_size):
        super(TrendsRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Map input to hidden to pass to rnn
        self.i2h = nn.Linear(input_size, hidden_size)

        # Apply multi-layer RNN
        self.rnn = RNN(hidden_size, 1, False)

        # Map hidden to output as a last step (after GRU cell finish)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward_inner(self, input_tensor, with_dropout=False):
        seq_length, batch_size, input_dim = input_tensor.size()

        input_tensor = torch.cat(list(map(lambda x: torch.unsqueeze(self.i2h(x), 0), input_tensor)), 0)

        output_hidden, hh = self.rnn(input_tensor)
        print(output_hidden.size())
        output_tensor = torch.cat(list(map(lambda x: torch.unsqueeze(self.h2o(x), 0), output_hidden)), 0)

        return torch.cat(output_tensor, dim=0)

    def forward(self, input_tensor):
        return self.forward_inner(input_tensor, with_dropout=True)

    def predict(self, input_tensor):
        return self.forward_inner(input_tensor, with_dropout=False)
