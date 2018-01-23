import torch
import torch.nn as nn
from torch.autograd import Variable


class TrendsGRU(nn.Module):
    """
    Network with embedded GRU cell.
    Architecture: GRU -> Linear
    
    https://en.wikipedia.org/wiki/Gated_recurrent_unit
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(TrendsGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.r = nn.Linear(input_size + hidden_size, hidden_size)

        # linear layer that will map (input, hidden * r) -> hidden
        self.h = nn.Linear(input_size + hidden_size, hidden_size)

        # Map hidden to linear as a last step (after GRU cell finish)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def single_forward(self, input_tensor, hidden, with_dropout=True):
        """
        Forward path for a one timestamp
        :return: output and hidden state
        """

        input_combined = torch.cat((input_tensor, hidden), 1)

        z = self.sigmoid(self.z(input_combined))
        r = self.sigmoid(self.r(input_combined))

        # combined for the h layer
        input_hr_combined = torch.cat((input_tensor, hidden * r), 1)

        # update hidden layer
        hidden = z * hidden + (1 - z) * self.sigmoid(self.h(input_hr_combined))

        output_tensor = self.h2o(hidden)

        if with_dropout:
            output_tensor = self.dropout(output_tensor)

        return output_tensor, hidden

    def forward_inner(self, input_tensor, with_dropout):
        seq_length, batch_size, input_dim = input_tensor.size()

        hidden = self.init_hidden(batch_size)
        output_all = []
        for i in range(seq_length):
            # do forward path for single timestamp
            out_combined, hidden = self.single_forward(input_tensor[i], hidden, with_dropout)

            # append output for current timestamp, view is necessary in order to be able to concat latter
            output_all.append(out_combined.view(1, batch_size, -1))
        return torch.cat(output_all, dim=0)

    def forward(self, input_tensor):
        return self.forward_inner(input_tensor, with_dropout=True)

    def predict(self, input_tensor):
        return self.forward_inner(input_tensor, with_dropout=False)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True)
