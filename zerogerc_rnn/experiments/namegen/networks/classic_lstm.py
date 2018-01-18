import torch
import torch.autograd as autograd
import torch.nn as nn

from zerogerc_rnn.experiments.namegen.reader import names_data_reader


class LSTM(nn.Module):
    """
    Network that uses classical LSTM architecture from official pyTorch docs.
    
    @link http://pytorch.org/docs/master/nn.html#torch.nn.LSTM
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i = nn.Linear(input_size + hidden_size, hidden_size)
        self.f = nn.Linear(input_size + hidden_size, hidden_size)
        self.g = nn.Linear(input_size + hidden_size, hidden_size)
        self.o = nn.Linear(input_size + hidden_size, hidden_size)

        # last layer that converts lstm out to desired output_size
        self.h2o = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()

    def single_forward(self, input, hidden, cell, with_dropout=True):
        """
        Forward path for a one timestamp
        :return: output and hidden state
        """

        input_combined = torch.cat((input, hidden), 1)

        # calc inner layers
        i = self.sigmoid(self.i(input_combined))
        f = self.sigmoid(self.f(input_combined))
        g = self.tanh(self.g(input_combined))
        o = self.sigmoid(self.o(input_combined))

        # recalculate cell and hidden state
        cell = f * cell + i * g
        hidden = o * self.tanh(cell)

        # convert hidden -> output
        output = self.h2o(hidden)

        # apply dropout if necessary, typically it would be disabled during prediction time
        if with_dropout:
            output = self.dropout(output)

        # softmax is necessary to be able to get letter as max of output
        output = self.softmax(output)

        return output, hidden, cell

    def forward(self, input_tensor):
        seq_length, batch_size, input_dim = input_tensor.size()

        hidden = self.init_hidden(batch_size)
        cell = self.init_cell(batch_size)

        output_all = []
        for i in range(seq_length):
            # do forward path for single timestamp
            out_combined, hidden, cell = self.single_forward(input_tensor[i], hidden, cell)

            # append output for current timestamp, view is necessary in order to be able to concat latter
            output_all.append(out_combined.view(1, batch_size, -1))
        return torch.cat(output_all, dim=0)

    def predict(self, input_tensor):
        seq_length, batch_size, input_dim = input_tensor.size()

        hidden = self.init_hidden(batch_size)
        cell = self.init_cell(batch_size)

        output_all = []
        for i in range(seq_length):
            if i == 0:  # first letter is an input on the first timestamp
                cur_input = input_tensor[i]
            else:  # input is the output of the previous iteration
                cur_input = names_data_reader.output_to_input_with_category(
                    category_tensor=input_tensor[i, :, :names_data_reader.n_categories],
                    output=output_all[-1].view(1, -1)
                )

            if cur_input is None:  # ES reached
                break

            # do forward path for single timestamp
            out_combined, hidden, cell = self.single_forward(cur_input, hidden, cell, with_dropout=False)

            # append output for current timestamp
            # view is necessary in order to be able to concat latter
            output_all.append(out_combined.view(1, batch_size, -1))
        return torch.cat(output_all, dim=0)

    def init_hidden(self, batch_size):
        return autograd.Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True)

    def init_cell(self, batch_size):
        return autograd.Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True)
