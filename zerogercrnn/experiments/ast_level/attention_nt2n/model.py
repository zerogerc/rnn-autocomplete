from itertools import chain

import torch
import torch.nn as nn

from zerogercrnn.experiments.utils import forget_hidden_partly_lstm_cell, repackage_hidden
from zerogercrnn.lib.core import PretrainedEmbeddingsModule, EmbeddingsModule, LSTMCellDropout, \
    LogSoftmaxOutputLayer
from zerogercrnn.lib.embedding import Embeddings


class NT2NAttentionModel(nn.Module):
    def __init__(
            self,
            seq_len,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminal_embeddings: Embeddings,
            hidden_dim,
            prediction_dim,
            dropout
    ):
        super().__init__()

        self.non_terminals_num = non_terminals_num
        self.non_terminal_embedding_dim = non_terminal_embedding_dim
        self.hidden_dim = hidden_dim
        self.prediction_dim = prediction_dim
        self.dropout = dropout

        self.nt_embedding = EmbeddingsModule(
            num_embeddings=self.non_terminals_num,
            embedding_dim=self.non_terminal_embedding_dim,
            sparse=False
        )

        self.t_embedding = PretrainedEmbeddingsModule(
            embeddings=terminal_embeddings,
            requires_grad=False,
            sparse=False
        )
        self.terminal_embedding_dim = self.t_embedding.embedding_dim

        self.recurrent_core = LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        )

        # self.attention = ContextBaseTailAttention(
        #     seq_len=seq_len,  # TODO: better way
        #     hidden_size=self.hidden_dim
        # )

        self.h2o = LogSoftmaxOutputLayer(
            input_size=self.hidden_dim,
            output_size=self.prediction_dim,
            dim=2
        )

    def parameters(self):
        return chain(self.nt_embedding.parameters(), self.t_embedding.parameters(), self.recurrent_core.parameters(),
                     self.h2o.parameters())

    def sparse_parameters(self):
        return chain(self.nt_embedding.sparse_parameters(), self.t_embedding.sparse_parameters(),
                     self.recurrent_core.sparse_parameters(),
                     self.h2o.sparse_parameters())

    def forward(self, non_terminal_input, terminal_input, hidden, forget_vector):
        assert non_terminal_input.size() == terminal_input.size()
        assert non_terminal_input.size() == terminal_input.size()

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)
        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        hidden = repackage_hidden(hidden)
        hidden = forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector)
        # self.attention.forget_context_partly(forget_vector=forget_vector)

        recurrent_output = []
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0
            cur_h, cur_c = self.recurrent_core(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            # cur_o = self.attention(cur_h)

            hidden = (cur_h, cur_c)
            recurrent_output.append(cur_h)  # torch.cat((cur_h, cur_o), dim=1)

        recurrent_output = torch.stack(recurrent_output, dim=0)
        prediction = self.h2o(recurrent_output)

        assert hidden is not None
        return prediction, hidden

    def init_hidden(self, batch_size, cuda, no_grad=False):
        # self.attention.init_hidden(batch_size, cuda, no_grad)
        return self.recurrent_core.init_hidden(batch_size, cuda, no_grad=no_grad)
