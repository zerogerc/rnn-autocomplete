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


class LayeredAttentionRecurrent(LayeredRecurrentUpdateAfter):

    def pick_current_output(self, layered_hidden, nodes_depth):
        return None


class NT2NBaseAttentionPlusLayeredModel(CombinedModule):
    """Base Model with attention on last n hidden states of LSTM."""

    def __init__(
            self,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminals_num,
            terminal_embedding_dim,
            hidden_dim,
            layered_hidden_size,
            num_layers,
            dropout
    ):
        super().__init__()

        self.non_terminals_num = non_terminals_num
        self.non_terminal_embedding_dim = non_terminal_embedding_dim
        self.terminals_num = terminals_num
        self.terminal_embedding_dim = terminal_embedding_dim
        self.hidden_dim = hidden_dim
        self.layered_hidden_size = layered_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_tree_layers = 50

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

        self.layered_recurrent = self.module(LayeredRecurrentUpdateAfter(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            num_tree_layers=self.num_tree_layers,
            single_hidden_size=self.layered_hidden_size
        ))

        self.h2o = self.module(LinearLayer(
            input_size=2 * self.hidden_dim + self.layered_hidden_size,
            output_size=self.non_terminals_num
        ))

    def forward(self, m_input: ASTInput, c_hidden, forget_vector):
        hidden, layered_hidden = c_hidden

        non_terminal_input = m_input.non_terminals
        terminal_input = m_input.terminals

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)
        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        recurrent_output, new_hidden, attn_output, layered_output, new_layered_hidden = \
            self.get_recurrent_layers_outputs(
                combined_input=combined_input,
                nodes_depth=m_input.nodes_depth,
                hidden=hidden,
                layered_hidden=layered_hidden,
                forget_vector=forget_vector
            )

        concatenated_output = torch.cat((recurrent_output, attn_output, layered_output), dim=-1)
        prediction = self.h2o(concatenated_output)

        return prediction, (new_hidden, new_layered_hidden)

    def get_recurrent_layers_outputs(
            self, combined_input, nodes_depth,
            hidden, layered_hidden, forget_vector
    ):
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
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0

            # core recurrent part
            cur_h, cur_c = self.recurrent_cell(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
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

        return recurrent_output, hidden, attn_output, layered_output, layered_hidden

    def init_hidden(self, batch_size):
        self.last_k_attention.init_hidden(batch_size)
        return self.recurrent_cell.init_hidden(batch_size), \
               self.layered_recurrent.init_hidden(batch_size)
