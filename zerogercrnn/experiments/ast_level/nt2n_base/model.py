import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.core import CombinedModule, EmbeddingsModule, RecurrentCore, LinearLayer
from zerogercrnn.lib.utils import repackage_hidden, forget_hidden_partly
from zerogercrnn.experiments.ast_level.ast_core import ASTNT2NModule


class NT2NBaseModel(ASTNT2NModule):
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
        super().__init__(
            non_terminals_num=non_terminals_num,
            non_terminal_embedding_dim=non_terminal_embedding_dim,
            terminals_num=terminals_num,
            terminal_embedding_dim=terminal_embedding_dim,
            recurrent_output_size=hidden_dim
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.recurrent_core = self.module(RecurrentCore(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type='lstm'
        ))

    def get_recurrent_output(self, combined_input, ast_input: ASTInput, m_hidden, forget_vector):
        hidden = m_hidden

        hidden = forget_hidden_partly(hidden, forget_vector=forget_vector)
        hidden = repackage_hidden(hidden)

        recurrent_output, new_hidden = self.recurrent_core(combined_input, hidden)

        return recurrent_output, new_hidden

    def init_hidden(self, batch_size):
        return self.recurrent_core.init_hidden(batch_size)
