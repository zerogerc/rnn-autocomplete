import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.lib.utils.time import logger

from zerogercrnn.experiments.js.ast_level.model.core import RecurrentCore
from zerogercrnn.experiments.js.ast_level.model.utils import init_uniform_layers

"""Base rnn model for JS AST prediction (in use since 01Apr till 01Apr)"""


class NTModel(nn.Module):

    def __init__(
            self,
            non_terminal_vocab_size,
            terminal_vocab_size,
            embedding_size,
            recurrent_layer: RecurrentCore,
    ):
        super(NTModel, self).__init__()
        assert recurrent_layer.input_size == embedding_size

        self.recurrent_out_size = recurrent_layer.hidden_size
        self.embedding_size = embedding_size
        self.non_terminal_output_size = non_terminal_vocab_size
        self.terminal_output_size = terminal_vocab_size

        # PyTorch is not able to have one optimizer for sparse and non-sparse layers, so we should split parameters
        self.dense_params = []
        self.sparse_params = []

        # Layer that encodes one-hot vector of non-terminals (A)
        self.non_terminal_embedding = self.sparse_model(nn.Embedding(
            num_embeddings=non_terminal_vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        ))

        # Layer that encodes one-hot vector of terminals (B)
        self.terminal_embedding = self.sparse_model(nn.Embedding(
            num_embeddings=terminal_vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        ))

        # Recurrent layer that will have (A + B) as an input
        self.recurrent = self.dense_model(
            recurrent_layer
        )

        # Layer that transforms hidden state of recurrent layer into next non-terminal
        self.h2NT = self.dense_model(
            nn.Linear(self.recurrent_out_size, self.non_terminal_output_size)
        )
        self.softmaxNT = nn.LogSoftmax(dim=1)

        # Layer that transforms hidden state of recurrent layer into next terminal
        self.h2T = self.dense_model(
            nn.Linear(self.recurrent_out_size, self.terminal_output_size)
        )
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
        non_terminal_input = torch.squeeze(non_terminal_input, dim=2)
        terminal_input = torch.squeeze(terminal_input, dim=2)

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

    def sparse_model(self, model):
        self.sparse_params += model.parameters()
        return model

    def dense_model(self, model):
        self.dense_params += model.parameters()
        return model

    def _init_params_(self):
        init_uniform_layers(
            min_value=-0.05,
            max_value=0.05,
            layers=[
                self.non_terminal_embedding,
                self.terminal_embedding,
                self.h2NT,
                self.h2T
            ]
        )
