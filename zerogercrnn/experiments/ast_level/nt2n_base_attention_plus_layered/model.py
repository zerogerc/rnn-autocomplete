import torch
import numpy as np

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.attn import calc_attention_combination
from zerogercrnn.lib.core import CombinedModule, EmbeddingsModule, RecurrentCore, LinearLayer, LSTMCellDropout, \
    LayeredRecurrentUpdateAfter
from zerogercrnn.lib.utils import repackage_hidden, forget_hidden_partly, get_best_device, setup_tensor, \
    forget_hidden_partly_lstm_cell
from zerogercrnn.experiments.ast_level.ast_core import LastKAttention
from zerogercrnn.lib.attn import Attn
from zerogercrnn.experiments.ast_level.ast_core import ASTNT2NModule


class LayeredAttentionRecurrent(LayeredRecurrentUpdateAfter):

    def pick_current_output(self, layered_hidden, nodes_depth):
        return None


class NT2NBaseAttentionPlusLayeredModel(ASTNT2NModule):
    """Base Model with attention on last n hidden states of LSTM."""

    def __init__(
            self,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminals_num,
            terminal_embedding_dim,
            hidden_dim,
            layered_hidden_size,
            dropout
    ):
        super().__init__(
            non_terminals_num=non_terminals_num,
            non_terminal_embedding_dim=non_terminal_embedding_dim,
            terminals_num=terminals_num,
            terminal_embedding_dim=terminal_embedding_dim,
            recurrent_output_size=2 * hidden_dim + layered_hidden_size
        )

        self.hidden_dim = hidden_dim
        self.layered_hidden_size = layered_hidden_size
        self.dropout = dropout
        self.num_tree_layers = 50

        self.recurrent_cell = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))
        self.layered_attention = self.module(Attn(method='general', hidden_size=self.layered_hidden_size))

        self.last_k_attention = self.module(LastKAttention(
            hidden_size=self.hidden_dim,
            k=50
        ))

        self.layered_recurrent = self.module(LayeredAttentionRecurrent(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            num_tree_layers=self.num_tree_layers,
            single_hidden_size=self.layered_hidden_size
        ))

    def get_recurrent_output(self, combined_input, ast_input: ASTInput, m_hidden, forget_vector):
        hidden, layered_hidden = m_hidden
        nodes_depth = ast_input.nodes_depth

        # repackage hidden and forgot hidden if program file changed
        hidden = repackage_hidden(forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector))
        layered_hidden = LayeredRecurrentUpdateAfter.repackage_and_partly_forget_hidden(
            layered_hidden=layered_hidden,
            forget_vector=forget_vector
        )
        self.last_k_attention.repackage_and_forget_buffer_partly(forget_vector)

        # prepare node depths (store only self.num_tree_layers)
        nodes_depth = torch.clamp(nodes_depth, min=0, max=self.num_tree_layers - 1)

        recurrent_output = []
        attn_output = []
        layered_output = []
        b_h = None
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0

            # core recurrent part
            cur_h, cur_c = self.recurrent_cell(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
            b_h = hidden
            recurrent_output.append(cur_h)

            # attn part
            cur_attn_output = self.last_k_attention(cur_h)
            attn_output.append(cur_attn_output)

            # layered part
            l_h, l_c = self.layered_recurrent(
                combined_input[i],
                nodes_depth[i],
                layered_hidden=layered_hidden,
                reinit_dropout=reinit_dropout
            )

            layered_hidden = LayeredRecurrentUpdateAfter.update_layered_lstm_hidden(
                layered_hidden=layered_hidden,
                node_depths=nodes_depth[i],
                new_value=(l_h, l_c)
            )

            layered_output_coefficients = self.layered_attention(l_h, layered_hidden[0])
            cur_layered_output = calc_attention_combination(layered_output_coefficients, layered_hidden[0])
            layered_output.append(cur_layered_output)  # maybe cat?

        # combine outputs from different layers
        recurrent_output = torch.stack(recurrent_output, dim=0)
        attn_output = torch.stack(attn_output, dim=0)
        layered_output = torch.stack(layered_output, dim=0)

        assert b_h == hidden
        concatenated_output = torch.cat((recurrent_output, attn_output, layered_output), dim=-1)

        return concatenated_output, (hidden, layered_hidden)

    def init_hidden(self, batch_size):
        self.last_k_attention.init_hidden(batch_size)
        return self.recurrent_cell.init_hidden(batch_size), \
               self.layered_recurrent.init_hidden(batch_size)
