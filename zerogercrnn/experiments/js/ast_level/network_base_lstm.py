import torch
import torch.nn as nn

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

        # Recurrent layer that will have (A + B) as an input
        self.lstm = nn.LSTM(
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
        logger.log_time_ms('SQUEEZE')

        # this tensors will be the size of [batch_size, seq_len, embedding_dim]
        non_terminal_emb = self.non_terminal_embedding(non_terminal_input.permute(1, 0))
        terminal_emb = self.terminal_embedding(terminal_input.permute(1, 0))
        logger.log_time_ms('EMBEDDING')

        non_terminal_emb = non_terminal_emb.permute(1, 0, 2)
        terminal_emb = terminal_emb.permute(1, 0, 2)
        logger.log_time_ms('PERMUTE')

        lstm_input = non_terminal_emb + terminal_emb
        logger.log_time_ms('PLUS')

        # output_tensor will be the size of (seq_len, batch_size, hidden_size * num_directions)
        output_tensor, hidden = self.lstm(lstm_input)
        logger.log_time_ms('LSTM')

        output_tensor = output_tensor.view(-1, self.hidden_size)  # flatten tensor for linear transformation
        logger.log_time_ms('OUT_VIEW')

        # converting to pair of (N, T)
        non_terminal_output = self.h2NT(output_tensor)
        logger.log_time_ms('NT_OUT_H2NT')
        non_terminal_output = self.softmaxNT(non_terminal_output)
        logger.log_time_ms('NT_OUT_SOFTMAX')
        non_terminal_output = non_terminal_output.view(seq_len, batch_size, self.non_terminal_output_size)
        logger.log_time_ms('NT_OUT_VIEW')

        terminal_output = self.h2T(output_tensor)
        logger.log_time_ms('T_OUT_H2T')
        terminal_output = self.softmaxT(terminal_output)
        logger.log_time_ms('T_OUT_SOFTMAX')
        terminal_output = terminal_output.view(seq_len, batch_size, self.terminal_output_size)
        logger.log_time_ms('T_OUT_VIEW')

        return non_terminal_output, terminal_output
