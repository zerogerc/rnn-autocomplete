import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.calculation import select_layered_hidden, set_layered_hidden
from zerogercrnn.lib.core import EmbeddingsModule, PretrainedEmbeddingsModule, LSTMCellDropout, \
    LinearLayer, CombinedModule, BaseModule
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils import forget_hidden_partly_lstm_cell, repackage_hidden
from zerogercrnn.lib.utils import wrap_cuda_no_grad_variable


def select_layered_lstm_hidden(layered_hidden, node_depths):
    return select_layered_hidden(layered_hidden[0], node_depths).squeeze(1), \
           select_layered_hidden(layered_hidden[1], node_depths).squeeze(1)


def update_layered_lstm_hidden(layered_hidden, node_depths, new_value):
    set_layered_hidden(layered_hidden[0], node_depths, new_value[0])
    set_layered_hidden(layered_hidden[1], node_depths, new_value[1])


class LayeredRecurrent(BaseModule):
    def __init__(self, input_size, tree_layers, single_hidden_size, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.single_hidden_size = single_hidden_size
        self.tree_layers = tree_layers
        self.output_size = single_hidden_size * 3
        self.layered_recurrent = LSTMCellDropout(
            input_size=self.input_size,
            hidden_size=self.single_hidden_size,
            dropout=dropout
        )

    def repackage_hidden(self, layered_hidden, forget_vector):
        layered_hidden = forget_hidden_partly_lstm_cell(layered_hidden,
                                                        forget_vector=forget_vector.unsqueeze(
                                                            1))  # TODO: check that shit
        layered_hidden[0].detach_(), layered_hidden[1].detach_()
        return layered_hidden

    def pick_current_output(self, layered_hidden, nodes_depth):
        o_cur = select_layered_hidden(layered_hidden[0], torch.clamp(nodes_depth, min=0, max=self.tree_layers - 1))
        o_prev = select_layered_hidden(layered_hidden[0], torch.clamp(nodes_depth - 1, min=0, max=self.tree_layers - 1))
        o_next = select_layered_hidden(layered_hidden[0], torch.clamp(nodes_depth + 1, min=0, max=self.tree_layers - 1))
        return torch.cat((o_prev, o_cur, o_next), dim=-1).squeeze()

    def forward(self, m_input, nodes_depth, layered_hidden, forget_vector, reinit_dropout):
        nodes_depth = torch.clamp(nodes_depth, max=self.tree_layers - 1)
        l_h, l_c = select_layered_lstm_hidden(layered_hidden, nodes_depth)
        l_h, l_c = self.layered_recurrent(m_input, (l_h, l_c), reinit_dropout=reinit_dropout)
        update_layered_lstm_hidden(layered_hidden, nodes_depth, (l_h, l_c))

    def init_hidden(self, batch_size, cuda, no_grad=False):
        h = wrap_cuda_no_grad_variable(
            torch.zeros((batch_size, self.tree_layers, self.single_hidden_size)), cuda=cuda, no_grad=no_grad)
        c = wrap_cuda_no_grad_variable(
            torch.zeros((batch_size, self.tree_layers, self.single_hidden_size)), cuda=cuda, no_grad=no_grad)

        return h, c


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
            single_hidden_size=100,
            dropout=self.dropout
        ))

        self.recurrent_core = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))

        self.h2o = self.module(LinearLayer(
            input_size=self.layered_recurrent.output_size + self.hidden_dim,
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
        layered_hidden = self.layered_recurrent.repackage_hidden(layered_hidden, forget_vector=forget_vector)

        recurrent_output = []
        recurrent_layered_output = []
        sl = combined_input.size()[0]
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0
            self.layered_recurrent(
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


def test_select():
    batch_size = 5
    layers = 50
    hidden_size = 10

    node_depths = torch.LongTensor([0, 2, layers - 1, 2, 5])
    layered_hidden = torch.randn((batch_size, layers, hidden_size))

    selected = select_layered_hidden(layered_hidden, node_depths)

    for i in range(node_depths.size()[0]):
        print(torch.nonzero(selected[i][0] == layered_hidden[i][node_depths[i]]).size()[0] == hidden_size)
        assert torch.nonzero(selected[i][0] == layered_hidden[i][node_depths[i]]).size()[0] == hidden_size


def test_update():
    batch_size = 6
    layers = 50
    hidden_size = 10

    layered_hidden = torch.randn((batch_size, layers, hidden_size))
    node_depths = torch.LongTensor([0, 1, layers - 1, 2, 5, 1])
    updated = torch.randn((batch_size, hidden_size))

    old_hidden = layered_hidden.clone()
    set_layered_hidden(layered_hidden, node_depths, updated)

    res = torch.nonzero(old_hidden - layered_hidden).size()[0] == batch_size * hidden_size
    print(res)
    assert res
    for i in range(node_depths.size()[0]):
        res = torch.nonzero(layered_hidden[i][node_depths[i]] == updated[i]).size()[0] == hidden_size
        print(res)
        assert res


if __name__ == '__main__':
    test_select()
    test_update()
