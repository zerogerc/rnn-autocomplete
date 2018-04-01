import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.experiments.js.ast_level.model.utils import init_recurrent_layers


class RecurrentCore(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., model_type='gru'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type

        if model_type == 'gru':
            self.recurrent = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type == 'lstm':
            self.recurrent = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise Exception('Unknown model type: {}'.format(model_type))

        self._init_params_()

    def _init_params_(self):
        init_recurrent_layers(self.recurrent)

    def forward(self, input_tensor, hidden):
        output_tensor, hidden = self.recurrent(input_tensor, hidden)
        return output_tensor, hidden

    def init_hidden(self, batch_size, cuda):
        h = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size)))
        c = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size)))

        if cuda:
            h = h.cuda()
            c = c.cuda()

        if self.model_type == 'lstm':
            return h, c
        else:
            return h
