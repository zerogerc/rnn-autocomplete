import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import RNN

class TrendsRNN(nn.Module):
    """Network with multi-layer RNN."""

    def __init__(self, input_size, hidden_size, output_size):
        super(TrendsRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Apply multi-layer RNN
        self.rnn = RNN(input_size, hidden_size, num_layers=1, dropout=0.1)

        # Map hidden to output as a last step (after GRU cell finish)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward_inner(self, input_tensor):
        seq_length, batch_size, input_dim = input_tensor.size()

        output_hidden, hh = self.rnn(input_tensor)
        output_tensor = self.h2o(output_hidden.view(-1, self.hidden_size))
        return output_tensor.view(seq_length, batch_size, self.output_size)

    def forward(self, input_tensor):
        return self.forward_inner(input_tensor)

    def predict(self, input_tensor):
        return self.forward_inner(input_tensor)
