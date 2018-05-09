import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.core import PretrainedEmbeddingsModule, EmbeddingsModule, RecurrentCore, \
    LinearLayer, CombinedModule
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils import forget_hidden_partly, repackage_hidden


class NTN2TBaseModel(CombinedModule):
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

        self.nt_embedding = self.module(EmbeddingsModule(
            num_embeddings=self.non_terminals_num,
            embedding_dim=self.non_terminal_embedding_dim,
            sparse=False
        ))

        self.t_embedding = self.module(PretrainedEmbeddingsModule(
            embeddings=terminal_embeddings,
            requires_grad=False
        ))
        self.terminals_num = self.t_embedding.num_embeddings
        self.terminal_embedding_dim = self.t_embedding.embedding_dim

        self.recurrent_core = self.module(RecurrentCore(
            input_size=2 * self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type='lstm'
        ))

        self.h2t = self.module(LinearLayer(
            input_size=self.hidden_dim,
            output_size=self.terminals_num,
            bias=False
        ))

    def forward(self, m_input: ASTInput, hidden, forget_vector):
        non_terminal_input = m_input.non_terminals
        terminal_input = m_input.terminals
        current_non_terminal_input = m_input.current_non_terminals

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)
        cur_nt_embedded = self.nt_embedding(current_non_terminal_input)

        combined_input = torch.cat([nt_embedded, cur_nt_embedded, t_embedded], dim=2)

        hidden = repackage_hidden(hidden)
        hidden = forget_hidden_partly(hidden, forget_vector=forget_vector)
        recurrent_output, new_hidden = self.recurrent_core(combined_input, hidden)

        prediction = self.h2t(recurrent_output)

        return prediction, new_hidden

    def init_hidden(self, batch_size):
        return self.recurrent_core.init_hidden(batch_size)
