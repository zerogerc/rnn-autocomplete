import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.attn import Attn
from zerogercrnn.lib.calculation import select_layered_hidden, calc_attention_combination, set_layered_hidden
from zerogercrnn.lib.core import EmbeddingsModule, LSTMCellDropout, \
    LinearLayer, CombinedModule, LayeredRecurrent, NormalizationLayer, BaseModule
from zerogercrnn.lib.utils import forget_hidden_partly_lstm_cell, repackage_hidden, setup_tensor
from zerogercrnn.experiments.ast_level.metrics import LayeredNodeDepthsAttentionMetrics


def create_one_hot_depths(node_depths, layers_num):  # checked
    batch_size = node_depths.size()[0]
    depths_one_hot = node_depths.new(batch_size, layers_num)
    return depths_one_hot.zero_().scatter_(1, node_depths.unsqueeze(1), 1).float()


class LayeredAttention(CombinedModule):
    def __init__(self, input_size, num_tree_layers):
        super().__init__()
        self.input_size = input_size
        self.num_tree_layers = num_tree_layers
        # self.attention_metrics = LayeredNodeDepthsAttentionMetrics()
        self.attn = self.module(Attn(method='general', hidden_size=self.input_size))

    def forward(self, m_input, layered_hidden, nodes_depth_target):
        attn_output_coefficients = self.attn(m_input, layered_hidden)
        # self.attention_metrics.report(nodes_depth_target, attn_output_coefficients)
        attn_output = calc_attention_combination(attn_output_coefficients, layered_hidden)
        return attn_output

    def init_hidden(self, batch_size):
        return setup_tensor(torch.zeros((batch_size, self.num_tree_layers, self.input_size)))

    @staticmethod
    def update_layers(new_value, nodes_depth_target, layered_hidden):
        return set_layered_hidden(layered_hidden, nodes_depth_target, new_value)

    @staticmethod
    def repackage_and_partly_forget_hidden(layered_hidden, forget_vector):  # checked
        layered_hidden = layered_hidden.mul(forget_vector.unsqueeze(1))
        return repackage_hidden(layered_hidden)


class NT2NSingleLSTMLayeredAttentionModel(CombinedModule):
    def __init__(
            self,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminals_num,
            terminal_embedding_dim,
            hidden_dim,
            node_depths_embedding_dim,
            dropout
    ):
        super().__init__()
        print('NT2NSingleLSTMLayeredAttentionModel created!')

        self.non_terminals_num = non_terminals_num
        self.non_terminal_embedding_dim = non_terminal_embedding_dim
        self.terminals_num = terminals_num
        self.terminal_embedding_dim = terminal_embedding_dim
        self.hidden_dim = hidden_dim
        self.node_depths_embedding_dim = node_depths_embedding_dim
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

        self.num_tree_layers = 50

        self.recurrent_core = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim + self.num_tree_layers,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))

        self.layered_attention = self.module(LayeredAttention(
            input_size=self.hidden_dim,
            num_tree_layers=self.num_tree_layers
        ))

        self.h2o = self.module(LinearLayer(
            input_size=2 * self.hidden_dim,
            output_size=self.non_terminals_num
        ))

    def forward(self, m_input: ASTInput, c_hidden, forget_vector):
        hidden, layered_hidden = c_hidden

        nt_embedded = self.nt_embedding(m_input.non_terminals)
        t_embedded = self.t_embedding(m_input.terminals)
        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        recurrent_output, hidden, layered_attn_output, layered_hidden = self.get_recurrent_layers_outputs(
            combined_input=combined_input,
            node_depths=m_input.nodes_depth,
            node_depths_target=m_input.nodes_depth_target,
            hidden=hidden,
            layered_hidden=layered_hidden,
            forget_vector=forget_vector
        )

        concat_output = torch.cat((recurrent_output, layered_attn_output), dim=-1)
        prediction = self.h2o(concat_output)
        return prediction, (hidden, layered_hidden)

    def get_recurrent_layers_outputs(
            self, combined_input, hidden, layered_hidden, forget_vector,
            node_depths, node_depths_target
    ):
        hidden = repackage_hidden(forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector))
        layered_hidden = LayeredAttention.repackage_and_partly_forget_hidden(
            layered_hidden=layered_hidden,
            forget_vector=forget_vector
        )

        node_depths = torch.clamp(node_depths, min=0, max=self.num_tree_layers - 1)
        node_depths_target = torch.clamp(node_depths_target, min=0, max=self.num_tree_layers - 1)

        recurrent_output = []
        layered_attn_output = []
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0

            # concat input with one-hot depth
            node_depths_one_hot = create_one_hot_depths(node_depths[i], self.num_tree_layers)
            recurrent_input = torch.cat((combined_input[i], node_depths_one_hot), dim=-1)

            # core recurrent part
            cur_h, cur_c = self.recurrent_core(recurrent_input, hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
            recurrent_output.append(cur_h)

            # layered part
            attn_output = self.layered_attention(cur_h, layered_hidden, node_depths_target[i])
            layered_attn_output.append(attn_output)

            # update according to known depths
            layered_hidden = LayeredAttention.update_layers(cur_h, node_depths_target[i], layered_hidden)

        # combine outputs from different layers
        recurrent_output = torch.stack(recurrent_output, dim=0)
        layered_attn_output = torch.stack(layered_attn_output, dim=0)

        return recurrent_output, hidden, layered_attn_output, layered_hidden

    def init_hidden(self, batch_size):
        return self.recurrent_core.init_hidden(batch_size), \
               self.layered_attention.init_hidden(batch_size)
