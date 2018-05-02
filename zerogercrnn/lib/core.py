from abc import abstractmethod
from itertools import chain

import torch
from torch import nn as nn

from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils import init_layers_uniform, init_recurrent_layers, setup_tensor, get_device


class HealthCheck:
    """Class that do some check on the model. Usually it prints some info about model at the end of epoch."""

    @abstractmethod
    def do_check(self):
        pass


class BaseModule(nn.Module):

    def sparse_parameters(self):  # in general modules do not care about sparse parameters.
        return []

    def health_checks(self):
        return []


class CombinedModule(BaseModule):
    """Module that contain sparse and non-sparse parameters."""

    def __init__(self):
        super().__init__()
        self.params = []
        self.modules = []

    def param(self, param):
        self.params.append(param)

    def module(self, module):
        self.modules.append(module)
        return module

    def parameters(self):
        return chain(self.params, *[m.parameters() for m in self.modules])

    def sparse_parameters(self):
        return chain(*[m.sparse_parameters() for m in self.modules])

    def health_checks(self):
        return chain(*[m.health_checks() for m in self.modules])


class PretrainedEmbeddingsModule(BaseModule):

    def __init__(self, embeddings: Embeddings, requires_grad=False, sparse=False):
        super().__init__()
        self.sparse = sparse

        self.num_embeddings = embeddings.embeddings_tensor.size()[0]
        self.embedding_dim = embeddings.embeddings_tensor.size()[1]

        self.embed = nn.Embedding(
            num_embeddings=embeddings.embeddings_tensor.size()[0],
            embedding_dim=embeddings.embeddings_tensor.size()[1],
            sparse=sparse
        )

        self.embed.weight.data.copy_(embeddings.embeddings_tensor)
        self.embed.weight.requires_grad = requires_grad

    def parameters(self):
        if self.sparse:
            return []
        else:
            return self.embed.parameters()

    def sparse_parameters(self):
        if self.sparse:
            return self.embed.parameters()
        else:
            return []

    def forward(self, model_input):
        return self.embed(model_input)


class EmbeddingsModule(BaseModule):
    def __init__(self, num_embeddings, embedding_dim, sparse=False):
        super().__init__()
        self.sparse = sparse

        self.model = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparse=sparse
        )

        init_layers_uniform(
            min_value=-0.1,
            max_value=0.1,
            layers=[self.model]
        )

    def parameters(self):
        if self.sparse:
            return []
        else:
            return self.model.parameters()

    def sparse_parameters(self):
        if self.sparse:
            return self.model.parameters()
        else:
            return []

    def forward(self, model_input):
        return self.model(model_input)


class RecurrentCore(BaseModule):

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

        init_recurrent_layers(self.recurrent)

    def forward(self, input_tensor, hidden):
        output_tensor, hidden = self.recurrent(input_tensor, hidden)
        return output_tensor, hidden

    def init_hidden(self, batch_size, cuda):
        h = setup_tensor(torch.zeros((self.num_layers, batch_size, self.hidden_size)), cuda)
        if self.model_type == 'lstm':
            c = setup_tensor(torch.zeros((self.num_layers, batch_size, self.hidden_size)), cuda)
            return h, c
        else:
            return h


def create_lstm_cell_hidden(hidden_size, batch_size, cuda):
    h = setup_tensor(torch.zeros((batch_size, hidden_size)), cuda=cuda)
    c = setup_tensor(torch.zeros((batch_size, hidden_size)), cuda=cuda)
    return h, c


class LSTMCellDropout(BaseModule):
    """Wrapper for **torch.nn.LSTMCell** that applies the same dropout to all timestamps.
    You could pass **reinit_dropout=True** to forward in order to reinit dropout mask.

    i.e. You reinit dropout after all sequence is processed.
    """

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.dropout_mask = None

    def forward(self, input_tensor, hidden_state, reinit_dropout=False):
        if (self.dropout_mask is None) or reinit_dropout:
            self._reinit_dropout_mask(input_tensor.size()[0], input_tensor.is_cuda)

        hidden, cell = self.lstm_cell(input_tensor, hidden_state)
        if self.training:
            return hidden * self.dropout_mask * (1. / (1. - self.dropout)), cell
        else:
            return hidden, cell

    def init_hidden(self, batch_size, cuda, no_grad=False):
        return create_lstm_cell_hidden(self.hidden_size, batch_size, cuda)

    def _reinit_dropout_mask(self, batch_size, cuda):
        if self.dropout_mask is None:
            tensor = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=get_device(cuda))
        else:
            tensor = self.dropout_mask

        # 1 - self.dropout, if dropout is 0.25 then probability to draw one would be 0.75
        self.dropout_mask = torch.bernoulli(tensor.fill_(1 - self.dropout))


class LinearLayer(BaseModule):
    """Layer that applies affine transformation and then LogSoftmax."""

    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.affine = nn.Linear(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=bias
        )

        init_layers_uniform(
            min_value=-0.05,
            max_value=0.05,
            layers=[
                self.affine
            ]
        )

    def forward(self, input):
        return self.affine(input)


class LogSoftmaxOutputLayer(BaseModule):
    """Layer that applies affine transformation and then LogSoftmax."""

    def __init__(self, input_size, output_size, dim, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dim = dim

        self.affine = nn.Linear(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=bias
        )

        self.log_softmax = nn.LogSoftmax(dim=dim)

        init_layers_uniform(
            min_value=-0.05,
            max_value=0.05,
            layers=[
                self.affine
            ]
        )

    def forward(self, input):
        return self.log_softmax(self.affine(input))


class PSumLayer(BaseModule):
    def __init__(self):
        super().__init__()
        self.mult_p = nn.Parameter(torch.randn(1))
        nn.init.uniform(self.mult_p, 0, 1)

    def forward(self, first_tensor, second_tensor):
        assert first_tensor.size() == second_tensor.size()
        return self.mult_p * first_tensor + (1 - self.mult_p) * second_tensor


class AlphaBetaSumLayer(BaseModule):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.mult_alpha = nn.Parameter(torch.randn(1))
        self.mult_beta = nn.Parameter(torch.randn(1))

        nn.init.uniform(self.mult_alpha, min_value, max_value)
        nn.init.uniform(self.mult_beta, min_value, max_value)

    def forward(self, first_tensor, second_tensor):
        assert first_tensor.size() == second_tensor.size()
        return self.mult_alpha * first_tensor + self.mult_beta * second_tensor

    def health_checks(self):
        return [AlphaBetaSumHealthCheck(self)]


class AlphaBetaSumHealthCheck(HealthCheck):

    def __init__(self, module: AlphaBetaSumLayer):
        super().__init__()
        self.module = module

    def do_check(self):
        print('Alpha: {}'.format(self.module.mult_alpha))
        print('Beta: {}'.format(self.module.mult_beta))
