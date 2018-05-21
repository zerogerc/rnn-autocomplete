import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.attn import Attn
from zerogercrnn.lib.calculation import select_layered_hidden, calc_attention_combination
from zerogercrnn.lib.core import EmbeddingsModule, LSTMCellDropout, \
    LinearLayer, CombinedModule, LayeredRecurrent, NormalizationLayer
from zerogercrnn.lib.utils import forget_hidden_partly_lstm_cell, repackage_hidden


class LayeredAttentionRecurrent(LayeredRecurrent):

    def pick_current_output(self, layered_hidden, nodes_depth):
        o_cur = select_layered_hidden(layered_hidden[0], torch.clamp(nodes_depth, min=0, max=self.num_tree_layers - 1))
        return o_cur.squeeze()


class NT2NLayeredAttentionNormalizedModel(CombinedModule):
    def __init__(
            self,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminals_num,
            terminal_embedding_dim,
            hidden_dim,
            layered_hidden_size,
            node_depths_embedding_dim,
            dropout
    ):
        super().__init__()
        print('NT2NLayeredAttentionNormalizedModel created!')

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
        self.input_norm = self.module(
            NormalizationLayer(features_num=self.non_terminal_embedding_dim + self.terminal_embedding_dim)
        )

        self.num_tree_layers = 50
        self.layered_hidden_size = layered_hidden_size
        self.layered_recurrent = self.module(LayeredAttentionRecurrent(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            num_tree_layers=self.num_tree_layers,
            single_hidden_size=self.layered_hidden_size,
            dropout=self.dropout
        ))

        self.attn = self.module(Attn(method='general', hidden_size=self.layered_hidden_size))

        self.recurrent_core = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))

        self.h_norm = self.module(NormalizationLayer(features_num=self.hidden_dim + self.layered_hidden_size))
        # self.norm = torch.nn.BatchNorm1d(self.hidden_dim + self.layered_hidden_size)

        self.h2o = self.module(LinearLayer(
            input_size=self.layered_hidden_size + self.hidden_dim,
            output_size=self.non_terminals_num
        ))

    def forward(self, m_input: ASTInput, c_hidden, forget_vector):
        hidden, layered_hidden = c_hidden

        nt_embedded = self.nt_embedding(m_input.non_terminals)
        t_embedded = self.t_embedding(m_input.terminals)
        concat_input = torch.cat([nt_embedded, t_embedded], dim=2)
        concat_input = self.input_norm(concat_input)

        recurrent_output, hidden, recurrent_layered_output, layered_hidden = self.get_recurrent_layers_outputs(
            combined_input=concat_input,
            node_depths=m_input.nodes_depth,
            hidden=hidden,
            layered_hidden=layered_hidden,
            forget_vector=forget_vector
        )

        concat_output = torch.cat((recurrent_output, recurrent_layered_output), dim=-1)
        concat_output = self.h_norm(concat_output)
        # sizes = concat_output.size()
        # concat_output = self.norm(concat_output.view(-1, sizes[-1])).view(sizes)

        prediction = self.h2o(concat_output)

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
                combined_input[i],
                node_depths[i],
                layered_hidden=layered_hidden,
                reinit_dropout=reinit_dropout
            )
            current_layered = self.layered_recurrent.pick_current_output(layered_hidden, node_depths[i])

            # layered attention part
            layered_output_coefficients = self.attn(current_layered, layered_hidden[0])
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
