import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.core import CombinedModule, EmbeddingsModule, RecurrentCore, LinearLayer
from zerogercrnn.lib.utils import repackage_hidden, forget_hidden_partly


class NT2NBaseModel(CombinedModule):
    def __init__(
            self,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminals_num,
            terminal_embedding_dim,
            hidden_dim,
            num_layers,
            dropout
    ):
        super().__init__()

        self.non_terminals_num = non_terminals_num
        self.non_terminal_embedding_dim = non_terminal_embedding_dim
        self.terminals_num = terminals_num
        self.terminal_embedding_dim = terminal_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.nt_embedding = self.module(EmbeddingsModule(
            num_embeddings=self.non_terminals_num,
            embedding_dim=self.non_terminal_embedding_dim,
            sparse=True
        ))

        self.t_embedding = self.module(EmbeddingsModule(
            num_embeddings=self.terminals_num,
            embedding_dim=self.terminal_embedding_dim,
            sparse=True
        ))

        self.recurrent_core = self.module(RecurrentCore(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type='lstm'
        ))

        self.h2o = self.module(LinearLayer(
            input_size=self.hidden_dim,
            output_size=self.non_terminals_num
        ))

    def forward(self, m_input: ASTInput, hidden, forget_vector):
        non_terminal_input = m_input.non_terminals
        terminal_input = m_input.terminals

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)

        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        hidden = repackage_hidden(hidden)
        hidden = forget_hidden_partly(hidden, forget_vector=forget_vector)
        recurrent_output, new_hidden = self.recurrent_core(combined_input, hidden)

        prediction = self.h2o(recurrent_output)

        return prediction, new_hidden

    def init_hidden(self, batch_size):
        return self.recurrent_core.init_hidden(batch_size)
