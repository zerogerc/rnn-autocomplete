import torch
import torch.nn.functional as F

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.attn import ContextAttention
from zerogercrnn.lib.core import PretrainedEmbeddingsModule, EmbeddingsModule, LSTMCellDropout, \
    LinearLayer, CombinedModule
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils import forget_hidden_partly_lstm_cell, repackage_hidden


class NT2NTailAttentionModel(CombinedModule):
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

        self.attention = self.module(ContextAttention(
            context_len=50,  # last 50 for context
            hidden_size=self.hidden_dim
        ))

        self.h2o = self.module(LinearLayer(
            input_size=2 * self.hidden_dim,
            output_size=self.prediction_dim
        ))

    def forward(self, m_input: ASTInput, hidden, forget_vector):
        non_terminal_input = m_input.non_terminals
        terminal_input = m_input.terminals

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)
        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        hidden = forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector)
        hidden = repackage_hidden(hidden)
        self.attention.forget_context_partly(forget_vector=forget_vector)

        recurrent_output = []
        sl = combined_input.size()[0]
        self.attention.train()
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0
            # if (i + 10 > sl) and self.training:
            #     self.attention.train()
            cur_h, cur_c = self.recurrent_core(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            # if i + 2 > sl and self.training:
            cur_o = self.attention(cur_h)
            # else:
            #     cur_o = cur_h

            hidden = (cur_h, cur_c)
            recurrent_output.append(F.relu(torch.cat((cur_h, cur_o), dim=1)))  # activation was added on 23Apr

        recurrent_output = torch.stack(recurrent_output, dim=0)
        prediction = self.h2o(recurrent_output)

        assert hidden is not None
        return prediction, hidden

    def init_hidden(self, batch_size):
        self.attention.init_hidden(batch_size)
        return self.recurrent_core.init_hidden(batch_size)
