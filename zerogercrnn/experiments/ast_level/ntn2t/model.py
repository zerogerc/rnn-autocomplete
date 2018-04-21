import torch
import torch.nn as nn
from itertools import chain

from zerogercrnn.lib.utils import forget_hidden_partly, repackage_hidden
from zerogercrnn.lib.core import PretrainedEmbeddingsModule, EmbeddingsModule, RecurrentCore, \
    LinearLayer
from zerogercrnn.lib.embedding import Embeddings


class NTN2TBaseModel(nn.Module):
    def __init__(
            self,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminal_embeddings: Embeddings,
            hidden_dim,
            num_layers,
            dropout
    ):
        super().__init__()

        self.non_terminals_num = non_terminals_num
        self.non_terminal_embedding_dim = non_terminal_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.nt_embedding = EmbeddingsModule(
            num_embeddings=self.non_terminals_num,
            embedding_dim=self.non_terminal_embedding_dim,
            sparse=False
        )

        self.t_embedding = PretrainedEmbeddingsModule(
            embeddings=terminal_embeddings,
            requires_grad=False
        )
        self.terminals_num = self.t_embedding.num_embeddings
        self.terminal_embedding_dim = self.t_embedding.embedding_dim

        self.recurrent_core = RecurrentCore(
            input_size=2 * self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type='lstm'
        )

        self.h2t = LinearLayer(
            input_size=self.hidden_dim,
            output_size=self.terminals_num,
            bias=False
        )

    def parameters(self):
        return chain(self.nt_embedding.parameters(), self.t_embedding.parameters(), self.recurrent_core.parameters(),
                     self.h2t.parameters())

    def sparse_parameters(self):
        return chain(self.nt_embedding.sparse_parameters(), self.t_embedding.sparse_parameters(),
                     self.recurrent_core.sparse_parameters(), self.h2t.sparse_parameters())

    def forward(self, non_terminal_input, terminal_input, current_non_terminal_input, hidden, forget_vector):
        assert non_terminal_input.size() == terminal_input.size()
        assert terminal_input.size() == current_non_terminal_input.size()

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)
        cur_nt_embedded = self.nt_embedding(current_non_terminal_input)

        combined_input = torch.cat([nt_embedded, cur_nt_embedded, t_embedded], dim=2)

        hidden = repackage_hidden(hidden)
        hidden = forget_hidden_partly(hidden, forget_vector=forget_vector)
        recurrent_output, new_hidden = self.recurrent_core(combined_input, hidden)

        prediction = self.h2t(recurrent_output)

        return prediction, new_hidden

    def init_hidden(self, batch_size, cuda, no_grad=False):
        return self.recurrent_core.init_hidden(batch_size, cuda, no_grad=no_grad)
