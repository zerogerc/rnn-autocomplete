import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.lib.utils.time import logger


class JSBaseModel(nn.Module):
    """Model that predicts next pair of (N, T) by sequence of (N, T)."""

    def __init__(
            self,
            non_terminal_vocab_size,
            terminal_vocab_size,
            embedding_size,
            hidden_size,
            num_layers=1,
            dropout=0.
    ):
        super(JSBaseModel, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
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

        self.lstm_hidden = None

        # Recurrent layer that will have (A + B) as an input
        self.lstm = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Layer that transforms hidden state of recurrent layer into next non-terminal
        self.h2NT = nn.Linear(self.hidden_size, self.non_terminal_output_size)
        self.softmaxNT = nn.LogSoftmax(dim=1)

        # Layer that transforms hidden state of recurrent layer into next terminal
        self.h2T = nn.Linear(self.hidden_size, self.terminal_output_size)
        self.softmaxT = nn.LogSoftmax(dim=1)

        self._init_params_()

    def sparse_params(self):
        for layer in [self.non_terminal_embedding, self.terminal_embedding]:
            for params in layer.parameters():
                yield params

    def non_sparse_params(self):
        for layer in [self.lstm, self.h2NT, self.h2T]:
            for params in layer.parameters():
                yield params

    def _init_params_(self):
        JSBaseModel._init_uniform_(
            min_value=-0.05,
            max_value=0.05,
            layers=[
                self.non_terminal_embedding,
                self.terminal_embedding,
                self.h2NT,
                self.h2T
            ]
        )

        JSBaseModel._init_lstm_(self.lstm)

    @staticmethod
    def _init_uniform_(min_value, max_value, layers):
        for layer in layers:
            for name, param in layer.named_parameters():
                nn.init.uniform(param, min_value, max_value)

    @staticmethod
    def _init_lstm_(*layers):
        for layer in layers:
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal(param)

    def forward(self, non_terminal_input, terminal_input):
        """
        :param input_tensor: tensor of size [seq_len, batch_size, 1]
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

        lstm_input = non_terminal_emb + terminal_emb
        logger.log_time_ms('PRE_IN')

        # output_tensor will be the size of (seq_len, batch_size, hidden_size * num_directions)
        output_tensor, hidden = self.lstm(lstm_input, self.lstm_hidden)
        logger.log_time_ms('RECURRENT')

        output_tensor = output_tensor.view(-1, self.hidden_size)  # flatten tensor for linear transformation

        # converting to pair of (N, T)
        non_terminal_output = self.h2NT(output_tensor)
        non_terminal_output = self.softmaxNT(non_terminal_output)
        non_terminal_output = non_terminal_output.view(seq_len, batch_size, self.non_terminal_output_size)

        terminal_output = self.h2T(output_tensor)
        terminal_output = self.softmaxT(terminal_output)
        terminal_output = terminal_output.view(seq_len, batch_size, self.terminal_output_size)
        logger.log_time_ms('PRE_OUT')

        return non_terminal_output, terminal_output

    def init_hidden(self, batch_size, cuda):
        h = Variable(torch.randn((self.num_layers, batch_size, self.hidden_size)))
        c = Variable(torch.randn((self.num_layers, batch_size, self.hidden_size)))

        if cuda:
            h = h.cuda()
            c = c.cuda()

        if isinstance(self.lstm, nn.LSTM):
            self.lstm_hidden = (h, c)
        else:
            self.lstm_hidden = h
