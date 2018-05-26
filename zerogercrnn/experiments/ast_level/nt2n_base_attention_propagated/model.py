import torch

from zerogercrnn.experiments.ast_level.ast_core import GatedLastKAttention, ASTNT2NModule
from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.core import LSTMCellDropout
from zerogercrnn.lib.utils import repackage_hidden, forget_hidden_partly_lstm_cell


class NT2NBaseAttentionPropaGatedBufferModel(ASTNT2NModule):
    """Model with gated attention on last n hidden states of LSTM."""

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
            recurrent_output_size=2 * hidden_dim
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.recurrent_cell = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))

        self.gated_attention = self.module(GatedLastKAttention(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            k=50
        ))

    def get_recurrent_output(self, combined_input, ast_input: ASTInput, m_hidden, forget_vector):
        hidden = repackage_hidden(forget_hidden_partly_lstm_cell(m_hidden, forget_vector=forget_vector))
        self.gated_attention.repackage_and_forget_buffer_partly(forget_vector)

        recurrent_output = []
        attn_output = []
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0

            # core recurrent part
            cur_h, cur_c = self.recurrent_cell(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
            recurrent_output.append(cur_h)

            # layered part
            cur_attn_output = self.gated_attention(combined_input[i], cur_h)
            attn_output.append(cur_attn_output)

            hidden = (cur_attn_output, cur_c)

        # combine outputs from different layers
        recurrent_output = torch.stack(recurrent_output, dim=0)
        attn_output = torch.stack(attn_output, dim=0)
        concatenated_output = torch.cat((recurrent_output, attn_output), dim=-1)

        return concatenated_output, hidden

    def init_hidden(self, batch_size):
        self.gated_attention.init_hidden(batch_size)
        return self.recurrent_cell.init_hidden(batch_size)
