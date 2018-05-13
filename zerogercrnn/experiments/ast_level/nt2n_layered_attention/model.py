import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.experiments.ast_level.metrics import LayeredNodeDepthsAttentionMetrics
from zerogercrnn.lib.attn import Attn
from zerogercrnn.lib.calculation import select_layered_hidden, calc_attention_combination
from zerogercrnn.lib.core import EmbeddingsModule, LSTMCellDropout, \
    LinearLayer, CombinedModule, LayeredRecurrent
from zerogercrnn.lib.utils import forget_hidden_partly_lstm_cell, repackage_hidden


class LayeredAttentionRecurrent(LayeredRecurrent):

    def pick_current_output(self, layered_hidden, nodes_depth):
        o_cur = select_layered_hidden(layered_hidden[0], torch.clamp(nodes_depth, min=0, max=self.tree_layers - 1))
        return o_cur.squeeze()


class NT2NLayeredAttentionModel(CombinedModule):
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
        super().__init__()

        self.non_terminals_num = non_terminals_num
        self.non_terminal_embedding_dim = non_terminal_embedding_dim
        self.terminals_num = terminals_num
        self.terminal_embedding_dim = terminal_embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.metric_node_depth_attn = self.additional_metrics[0]

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
        self.layered_hidden_size = layered_hidden_size
        self.layered_recurrent = self.module(LayeredAttentionRecurrent(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            tree_layers=self.num_tree_layers,
            single_hidden_size=self.layered_hidden_size,
            dropout=self.dropout
        ))

        self.attn = self.module(Attn(method='general', hidden_size=self.layered_hidden_size))

        self.recurrent_core = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))

        self.h2o = self.module(LinearLayer(
            input_size=self.layered_hidden_size + self.hidden_dim,
            output_size=self.non_terminals_num
        ))

    def create_additional_metrics(self):
        return [LayeredNodeDepthsAttentionMetrics()]

    def forward(self, m_input: ASTInput, c_hidden, forget_vector):
        hidden, layered_hidden = c_hidden

        nt_embedded = self.nt_embedding(m_input.non_terminals)
        t_embedded = self.t_embedding(m_input.terminals)

        recurrent_output, hidden, recurrent_layered_output, layered_hidden = self.get_recurrent_layers_outputs(
            combined_input=torch.cat([nt_embedded, t_embedded], dim=2),
            node_depths=m_input.nodes_depth,
            hidden=hidden,
            layered_hidden=layered_hidden,
            forget_vector=forget_vector
        )

        prediction = self.h2o(torch.cat((recurrent_output, recurrent_layered_output), dim=-1))

        assert hidden is not None
        return prediction, (hidden, layered_hidden)

    def get_recurrent_layers_outputs(self, combined_input, hidden, layered_hidden, forget_vector, node_depths):
        hidden = repackage_hidden(forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector))
        layered_hidden = LayeredRecurrent.repackage_and_partly_forget_hidden(
            layered_hidden=layered_hidden,
            forget_vector=forget_vector
        )

        recurrent_output = []
        recurrent_layered_output = []
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0

            # layered part
            layered_hidden = self.layered_recurrent(
                m_input=combined_input[i],
                nodes_depth=node_depths[i],
                layered_hidden=layered_hidden,
                reinit_dropout=reinit_dropout
            )
            current_layered = self.layered_recurrent.pick_current_output(layered_hidden, node_depths[i])

            # layered attention part
            layered_output_coefficients = self.attn(current_layered, layered_hidden[0])
            if self.report_to_additional_metrics:
                self.metric_node_depth_attn.report(node_depths[i], layered_output_coefficients)
            layered_output = calc_attention_combination(layered_output_coefficients, layered_hidden[0])
            recurrent_layered_output.append(layered_output)

            # core recurrent part
            cur_h, cur_c = self.recurrent_core(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
            recurrent_output.append(cur_h)

        recurrent_output = torch.stack(recurrent_output, dim=0)
        recurrent_layered_output = torch.stack(recurrent_layered_output, dim=0)

        return recurrent_output, hidden, recurrent_layered_output, layered_hidden

    def init_hidden(self, batch_size):
        return self.recurrent_core.init_hidden(batch_size), \
               self.layered_recurrent.init_hidden(batch_size)
