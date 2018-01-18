import torch
import torch.autograd as autograd
import torch.nn as nn

from zerogerc_rnn.experiments.namegen.reader import names_data_reader


class RNN(nn.Module):
    """
    Simple RNN model with one hidden layer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.input_size + self.hidden_size, self.output_size)
        self.o2o = nn.Linear(self.hidden_size + self.output_size, self.output_size)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.LogSoftmax()

    def single_forward(self, input_tensor, hidden, with_dropout=True):
        """
        Forward path for a one timestamp
        :return: output and hidden state
        """

        # combine input and hidden layer
        input_combined = torch.cat((input_tensor, hidden), 1)

        # convert to output and hidden with different layers
        output = self.i2o(input_combined)
        hidden = self.i2h(input_combined)

        # combine output and hidden
        out_combined = torch.cat((output, hidden), 1)

        # convert output + hidden -> output
        out_combined = self.o2o(out_combined)

        # apply dropout if necessary, typically it would be disabled during prediction time
        if with_dropout:
            out_combined = self.dropout(out_combined)

        # softmax is necessary to be able to get next letter as max of output
        out_combined = self.softmax(out_combined)
        return out_combined, hidden

    def forward(self, input_tensor):
        seq_length, batch_size, input_dim = input_tensor.size()
        hidden = self.init_hidden(batch_size)

        output_all = []
        for i in range(seq_length):
            # do forward path for single timestamp
            out_combined, hidden = self.single_forward(input_tensor[i], hidden)

            # append output for current timestamp, view is necessary in order to be able to concat latter
            output_all.append(out_combined.view(1, batch_size, -1))
        return torch.cat(output_all, dim=0)

    def predict(self, input_tensor):
        seq_length, batch_size, input_dim = input_tensor.size()
        hidden = self.init_hidden(batch_size)

        output_all = []
        for i in range(seq_length):
            if i == 0:  # first letter is an input
                cur_input = input_tensor[i]
            else:  # input is the output of the previous iteration
                cur_input = names_data_reader.output_to_input_with_category(
                    input_tensor[i, :, :names_data_reader.n_categories],
                    output_all[-1].view(1, -1)
                )

            if cur_input is None:  # ES reached
                break

            # do forward path for single timestamp
            out_combined, hidden = self.single_forward(cur_input, hidden, with_dropout=False)

            # append output for current timestamp, view is necessary in order to be able to concat latter
            output_all.append(out_combined.view(1, batch_size, -1))
        return torch.cat(output_all, dim=0)

    def init_hidden(self, batch_size):
        return autograd.Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True)
