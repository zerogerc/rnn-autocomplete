import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.lib.utils.time import logger


def _init_lstm_(*layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)


def _init_uniform_(min_value, max_value, layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            nn.init.uniform(param, min_value, max_value)


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

        self.hidden = None
        self._init_params_()

    def _init_params_(self):
        _init_lstm_(self.recurrent)

    def forward(self, input_tensor):
        output_tensor, hidden_tensor = self.recurrent(input_tensor)
        return output_tensor

    def init_hidden(self, batch_size, cuda):
        h = Variable(torch.randn((self.num_layers, batch_size, self.hidden_size)))
        c = Variable(torch.randn((self.num_layers, batch_size, self.hidden_size)))

        if cuda:
            h = h.cuda()
            c = c.cuda()

        if self.model_type == 'lstm':
            self.hidden = (h, c)
        else:
            self.hidden = h


class JSBaseModel(nn.Module):
    """Model that predicts next pair of (N, T) by sequence of (N, T).
    Core input size should be non_terminal_vocab_size + terminal_vocab_size
    """

    def __init__(
            self,
            non_terminal_vocab_size,
            terminal_vocab_size,
            embedding_size,
            recurrent_layer: RecurrentCore,
    ):
        super(JSBaseModel, self).__init__()
        assert recurrent_layer.input_size == embedding_size

        self.recurrent_out_size = recurrent_layer.hidden_size
        self.embedding_size = embedding_size
        self.non_terminal_output_size = non_terminal_vocab_size
        self.terminal_output_size = terminal_vocab_size

        # Layer that encodes one-hot vector of non-terminals (A)
        self.non_terminal_embedding = nn.Embedding(
            num_embeddings=non_terminal_vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        )

        # Layer that encodes one-hot vector of terminals (B)
        self.terminal_embedding = nn.Embedding(
            num_embeddings=terminal_vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        )

        # Recurrent layer that will have (A + B) as an input
        self.recurrent = recurrent_layer

        # Layer that transforms hidden state of recurrent layer into next non-terminal
        self.h2NT = nn.Linear(self.recurrent_out_size, self.non_terminal_output_size)
        self.softmaxNT = nn.LogSoftmax(dim=1)

        # Layer that transforms hidden state of recurrent layer into next terminal
        self.h2T = nn.Linear(self.recurrent_out_size, self.terminal_output_size)
        self.softmaxT = nn.LogSoftmax(dim=1)

        self._init_params_()

    def forward(self, non_terminal_input, terminal_input):
        """
        :param non_terminal_input: tensor of size [seq_len, batch_size, 1]
        :param terminal_input: tensor of size [seq_len, batch_size, 1]
        """
        assert non_terminal_input.size() == terminal_input.size()
        seq_len = non_terminal_input.size()[0]
        batch_size = non_terminal_input.size()[1]

        logger.reset_time()
        non_terminal_input = torch.squeeze(non_terminal_input)
        terminal_input = torch.squeeze(terminal_input)

        # this tensors will be the size of [batch_size, seq_len, embedding_dim]
        non_terminal_emb = self.non_terminal_embedding(non_terminal_input.permute(1, 0))
        terminal_emb = self.terminal_embedding(terminal_input.permute(1, 0))

        non_terminal_emb = non_terminal_emb.permute(1, 0, 2)
        terminal_emb = terminal_emb.permute(1, 0, 2)

        recurrent_input = non_terminal_emb + terminal_emb
        logger.log_time_ms('PRE_IN')

        # output_tensor will be the size of (seq_len, batch_size, hidden_size * num_directions)
        recurrent_output = self.recurrent(recurrent_input)
        logger.log_time_ms('RECURRENT')

        # flatten tensor for linear transformation
        recurrent_output = recurrent_output.view(-1, self.recurrent_out_size)

        # converting to pair of (N, T)
        non_terminal_output = self.h2NT(recurrent_output)
        non_terminal_output = self.softmaxNT(non_terminal_output)
        non_terminal_output = non_terminal_output.view(seq_len, batch_size, self.non_terminal_output_size)

        terminal_output = self.h2T(recurrent_output)
        terminal_output = self.softmaxT(terminal_output)
        terminal_output = terminal_output.view(seq_len, batch_size, self.terminal_output_size)
        logger.log_time_ms('PRE_OUT')

        return non_terminal_output, terminal_output

    def sparse_params(self):
        for layer in [self.non_terminal_embedding, self.terminal_embedding]:
            for params in layer.parameters():
                yield params

    def non_sparse_params(self):
        for layer in [self.lstm, self.h2NT, self.h2T]:
            for params in layer.parameters():
                yield params

    def _init_params_(self):
        _init_uniform_(
            min_value=-0.05,
            max_value=0.05,
            layers=[
                self.non_terminal_embedding,
                self.terminal_embedding,
                self.h2NT,
                self.h2T
            ]
        )
