import torch
import torch.nn.functional as F

from zerogercrnn.lib.core import PretrainedEmbeddingsModule, EmbeddingsModule, LSTMCellDropout, \
    LogSoftmaxOutputLayer, CombinedModule
from zerogercrnn.lib.attn import ContextAttention
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

        self.h2o = self.module(LogSoftmaxOutputLayer(
            input_size=2 * self.hidden_dim,
            output_size=self.prediction_dim,
            dim=2
        ))

    def forward(self, non_terminal_input, terminal_input, hidden, forget_vector):
        assert non_terminal_input.size() == terminal_input.size()
        assert non_terminal_input.size() == terminal_input.size()

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)
        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        hidden = forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector)
        hidden = repackage_hidden(hidden)
        self.attention.forget_context_partly(forget_vector=forget_vector)

        recurrent_output = []
        sl = combined_input.size()[0]
        self.attention.eval()
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0
            if (i + 10 > sl) and self.training:
                self.attention.train()
            cur_h, cur_c = self.recurrent_core(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            cur_o = self.attention(cur_h)

            if i + 10 <= sl:
                cur_h = cur_h.detach()
                cur_o = cur_o.detach()

            hidden = (cur_h, cur_c)
            recurrent_output.append(torch.cat((cur_h, F.sigmoid(cur_o)), dim=1)) # activation was added on 23Apr

        recurrent_output = torch.stack(recurrent_output, dim=0)
        prediction = self.h2o(recurrent_output)

        assert hidden is not None
        return prediction, hidden

    def init_hidden(self, batch_size, cuda, no_grad=False):
        self.attention.init_hidden(batch_size, cuda, no_grad)
        return self.recurrent_core.init_hidden(batch_size, cuda, no_grad=no_grad)
