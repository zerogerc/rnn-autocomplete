import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.core import EmbeddingsModule, PretrainedEmbeddingsModule, LSTMCellDropout, \
    LinearLayer, CombinedModule, BaseModule, create_lstm_cell_hidden
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils import forget_hidden_partly_lstm_cell, repackage_hidden


def select_hidden(layered_hidden, node_depths):
    return torch.stack([layered_hidden[node_depths[i]][0][i] for i in range(node_depths.size()[0])], dim=0), \
           torch.stack([layered_hidden[node_depths[i]][0][i] for i in range(node_depths.size()[0])], dim=0)


def update_hidden(layered_hidden, new_value, node_depths):
    for i in range(node_depths.size()[0]):
        layered_hidden[node_depths[i]][0][i] = new_value[0][i]
        layered_hidden[node_depths[i]][1][i] = new_value[1][i]


class LayeredRecurrent(BaseModule):
    def __init__(self, input_size, tree_layers, single_hidden_size):
        super().__init__()
        self.input_size = input_size
        self.single_hidden_size = single_hidden_size
        self.tree_layers = tree_layers
        self.output_size = single_hidden_size * 2
        self.layered_recurrent = LSTMCellDropout(
            input_size=self.input_size,
            hidden_size=self.single_hidden_size
        )

    def repackage_hidden(self, layered_hidden, forget_vector):
        for i in range(self.tree_layers):
            layered_hidden[i] = forget_hidden_partly_lstm_cell(layered_hidden[i], forget_vector=forget_vector)
            layered_hidden[i] = repackage_hidden(layered_hidden[i])

    def pick_current_output(self, layered_hidden, nodes_depth):
        nodes_depth = torch.clamp(nodes_depth, max=self.tree_layers - 1)
        l_h = [torch.cat((layered_hidden[nodes_depth[i]][0][i - 1], layered_hidden[nodes_depth[i]][0][i]), dim=-1) for i
               in range(nodes_depth.size()[0])]
        l_h = torch.stack(l_h, dim=0)
        return l_h

    def forward(self, m_input, nodes_depth, layered_hidden, forget_vector, reinit_dropout):
        nodes_depth = torch.clamp(nodes_depth, max=self.tree_layers - 1)
        l_h, l_c = select_hidden(layered_hidden, nodes_depth)
        l_h, l_c = self.layered_recurrent(m_input, (l_h, l_c), reinit_dropout=reinit_dropout)
        update_hidden(layered_hidden, (l_h, l_c), nodes_depth)
        return layered_hidden

    def init_hidden(self, batch_size, cuda, no_grad=False):
        return [create_lstm_cell_hidden(
            hidden_size=self.single_hidden_size,
            batch_size=batch_size,
            cuda=cuda,
            no_grad=no_grad
        ) for i in range(self.tree_layers)]


class NT2NLayerModel(CombinedModule):
    def __init__(
            self,
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

        self.layered_recurrent = self.module(LayeredRecurrent(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            tree_layers=50,
            single_hidden_size=30
        ))

        self.recurrent_core = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))

        self.h2o = self.module(LinearLayer(
            input_size=self.hidden_dim + self.layered_recurrent.output_size,
            output_size=self.prediction_dim
        ))

    def forward(self, m_input: ASTInput, c_hidden, forget_vector):
        hidden, layered_hidden = c_hidden

        nt_embedded = self.nt_embedding(m_input.non_terminals)
        t_embedded = self.t_embedding(m_input.terminals)
        node_depths = m_input.nodes_depth
        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        hidden = forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector)
        hidden = repackage_hidden(hidden)
        self.layered_recurrent.repackage_hidden(layered_hidden, forget_vector=forget_vector)

        recurrent_output = []
        recurrent_layered_output = []
        sl = combined_input.size()[0]
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0
            layered_hidden = self.layered_recurrent(
                m_input=combined_input[i],
                nodes_depth=node_depths[i],
                layered_hidden=layered_hidden,
                forget_vector=forget_vector,
                reinit_dropout=reinit_dropout
            )
            recurrent_layered_output.append(self.layered_recurrent.pick_current_output(layered_hidden, node_depths[i]))

            cur_h, cur_c = self.recurrent_core(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
            recurrent_output.append(cur_h)

        recurrent_output = torch.stack(recurrent_output, dim=0)
        recurrent_layered_output = torch.stack(recurrent_layered_output, dim=0)
        prediction = self.h2o(torch.cat((recurrent_output, recurrent_layered_output), dim=-1))

        assert hidden is not None
        return prediction, (hidden, layered_hidden)

    def init_hidden(self, batch_size, cuda, no_grad=False):
        return self.recurrent_core.init_hidden(batch_size, cuda, no_grad=no_grad), \
               self.layered_recurrent.init_hidden(batch_size, cuda, no_grad=no_grad)
