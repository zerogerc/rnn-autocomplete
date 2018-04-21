import torch
from torch import nn as nn
from torch.autograd import Variable

from zerogercrnn.experiments.utils import init_layers_uniform, init_recurrent_layers
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.experiments.utils import wrap_cuda_no_grad_variable
from zerogercrnn.lib.calculation import shift_left, calc_attention_combination, drop_matrix_rows_3d


class PretrainedEmbeddingsModule(nn.Module):

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


class EmbeddingsModule(nn.Module):
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

        init_recurrent_layers(self.recurrent)

    def parameters(self):
        return super().parameters()

    def sparse_parameters(self):
        return []

    def forward(self, input_tensor, hidden):
        output_tensor, hidden = self.recurrent(input_tensor, hidden)
        return output_tensor, hidden

    def init_hidden(self, batch_size, cuda, no_grad=False):
        if no_grad:
            h = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size)), volatile=True)
            c = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size)), volatile=True)
        else:
            h = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size)))
            c = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size)))

        if cuda:
            h = h.cuda()
            c = c.cuda()

        if self.model_type == 'lstm':
            return h, c
        else:
            return h


class LSTMCellDropout(nn.Module):
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

    def parameters(self):
        return super().parameters()

    def sparse_parameters(self):
        return []

    def forward(self, input_tensor, hidden_state, apply_dropout=True, reinit_dropout=False):
        if (self.dropout_mask is None) or reinit_dropout:
            self._reinit_dropout_mask(input_tensor.size()[0], input_tensor.is_cuda)

        hidden, cell = self.lstm_cell(input_tensor, hidden_state)
        if apply_dropout:
            return hidden * self.dropout_mask, cell
        else:
            return hidden, cell

    def init_hidden(self, batch_size, cuda, no_grad=False):
        h = wrap_cuda_no_grad_variable(torch.zeros((batch_size, self.hidden_size)), cuda=cuda, no_grad=no_grad)
        c = wrap_cuda_no_grad_variable(torch.zeros((batch_size, self.hidden_size)), cuda=cuda, no_grad=no_grad)
        return h, c

    def _reinit_dropout_mask(self, batch_size, cuda):
        if self.dropout_mask is None:
            if cuda:
                tensor = torch.cuda.FloatTensor(batch_size, self.hidden_size)
            else:
                tensor = torch.FloatTensor(batch_size, self.hidden_size)
        else:
            tensor = self.dropout_mask.data

        # 1 - self.dropout, if dropout is 0.25 then probability to draw one would be 0.75
        self.dropout_mask = Variable(torch.bernoulli(tensor.fill_(1 - self.dropout)))


class LinearLayer(nn.Module):
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

    def parameters(self):
        return super().parameters()

    def sparse_parameters(self):
        return []

    def forward(self, input):
        return self.affine(input)


class LogSoftmaxOutputLayer(nn.Module):
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

    def parameters(self):
        return super().parameters()

    def sparse_parameters(self):
        return []

    def forward(self, input):
        return self.log_softmax(self.affine(input))


class PSumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mult_p = nn.Parameter(torch.randn(1))
        nn.init.uniform(self.mult_p, 0, 1)

    def parameters(self):
        return super().parameters()

    def sparse_parameters(self):
        return []

    def forward(self, first_tensor, second_tensor):
        assert first_tensor.size() == second_tensor.size()
        return self.mult_p * first_tensor + (1 - self.mult_p) * second_tensor


class AlphaBetaSumLayer(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.mult_alpha = nn.Parameter(torch.randn(1))
        self.mult_beta = nn.Parameter(torch.randn(1))

        nn.init.uniform(self.mult_alpha, min_value, max_value)
        nn.init.uniform(self.mult_beta, min_value, max_value)

    def parameters(self):
        return super().parameters()

    def sparse_parameters(self):
        return []

    def forward(self, first_tensor, second_tensor):
        assert first_tensor.size() == second_tensor.size()
        return self.mult_alpha * first_tensor + self.mult_beta * second_tensor


class ContextBaseTailAttention(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.it = 0

        # Layer that applies attention to past self.cntx hidden states of contexts
        self.W = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.attn_softmax = nn.Softmax(dim=1)

        self.p_sum = PSumLayer()

        # Matrix that will hold past seq_len contexts. No backprop will be computed
        # size: [batch_size, seq_len, hidden_size]
        self.cntx_matrix = None

        nn.init.uniform(self.W, -0.05, 0.05)

    def init_hidden(self, batch_size, cuda, no_grad=False):
        self.cntx_matrix = wrap_cuda_no_grad_variable(
            torch.FloatTensor(batch_size, self.seq_len, self.hidden_size),
            cuda=cuda,
            no_grad=no_grad
        )

    def parameters(self):
        return super().parameters()

    def sparse_parameters(self):
        return []

    def forget_context_partly(self, forget_vector):
        """Method to drop context for programs that ended.
        :param forget_vector vector of size [batch_size, 1] with either 0 or 1
        """
        drop_matrix_rows_3d(self.cntx_matrix.data, forget_vector)

    def forward(self, h_t):
        """
        :param h_t: current hidden state of size [batch_size, hidden_size]
        :return: hidden state with applied sum attention of size [batch_size, hidden_size]
        """
        assert self.cntx_matrix is not None

        # self.cntx_matrix = self.cntx_matrix.detach()  # Just for sure =)

        # Calculating attention coefficients
        # * Make batch first and apply W
        attn_weights = self.cntx_matrix.matmul(self.W)
        # * Multiply by current hidden state to get coefficients
        attn_weights = attn_weights.matmul(h_t.unsqueeze(2))
        # * Softmax to make all of them sum to 1
        attn_weights = self.attn_softmax(attn_weights)
        self.it += 1
        if self.it % 10000 == 0:
            print(attn_weights.data[0])

        # Calc current context vector as sum of previous contexts multiplied by alpha
        cntx = calc_attention_combination(attn_weights, self.cntx_matrix)
        cntx = self.p_sum(h_t, cntx)

        shift_left(self.cntx_matrix.data, dimension=1)
        self.cntx_matrix.data[:, -1, :].copy_(h_t.data)

        return cntx
