import torch

from zerogercrnn.experiments.ast_level.data import ASTInput, ASTTarget
from zerogercrnn.lib.core import EmbeddingsModule, PretrainedEmbeddingsModule, LSTMCellDropout, \
    LinearLayer, CombinedModule
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils import forget_hidden_partly_lstm_cell, repackage_hidden


class NT2NSequentialModel(CombinedModule):
    def __init__(
            self,
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

        self.nt_embedding = self.module(EmbeddingsModule(
            num_embeddings=self.non_terminals_num,
            embedding_dim=self.non_terminal_embedding_dim,
            sparse=False
        ))

        self.t_embedding = self.module(PretrainedEmbeddingsModule(
            embeddings=terminal_embeddings,
            requires_grad=False,
            sparse=False
        ))

        self.terminal_embedding_dim = self.t_embedding.embedding_dim

        self.recurrent_core = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))

        self.h2o = self.module(LinearLayer(
            input_size=self.hidden_dim,
            output_size=self.prediction_dim
        ))

    def forward(self, m_input: ASTInput, hidden, forget_vector):
        nt_embedded = self.nt_embedding(m_input.non_terminals)
        t_embedded = self.t_embedding(m_input.terminals)
        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        hidden = forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector)
        hidden = repackage_hidden(hidden)

        recurrent_output = []
        sl = combined_input.size()[0]
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0
            cur_h, cur_c = self.recurrent_core(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
            recurrent_output.append(cur_h)

        recurrent_output = torch.stack(recurrent_output, dim=0)
        prediction = self.h2o(recurrent_output)

        assert hidden is not None
        return prediction, hidden

    def init_hidden(self, batch_size, cuda, no_grad=False):
        return self.recurrent_core.init_hidden(batch_size, cuda, no_grad=no_grad)
