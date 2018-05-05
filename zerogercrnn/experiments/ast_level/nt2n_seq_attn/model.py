import torch

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.attn import Attn
from zerogercrnn.lib.calculation import select_layered_hidden, set_layered_hidden, calc_attention_combination
from zerogercrnn.lib.core import EmbeddingsModule, PretrainedEmbeddingsModule, LSTMCellDropout, \
    LinearLayer, CombinedModule, BaseModule
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils import forget_hidden_partly_lstm_cell, repackage_hidden
from zerogercrnn.lib.utils import setup_tensor, get_device


def create_one_hot_depths(node_depths, batch_size, layers_num):
    depths_one_hot = node_depths.new(batch_size, layers_num)
    return depths_one_hot.zero_().scatter_(1, node_depths.unsqueeze(1), 1).float()


def select_layered_lstm_hidden(layered_hidden, node_depths):
    return select_layered_hidden(layered_hidden[0], node_depths).squeeze(1), \
           select_layered_hidden(layered_hidden[1], node_depths).squeeze(1)


def update_layered_lstm_hidden(layered_hidden, node_depths, new_value):
    return set_layered_hidden(layered_hidden[0], node_depths, new_value[0]), \
           set_layered_hidden(layered_hidden[1], node_depths, new_value[1])


class LayeredRecurrent(BaseModule):
    def __init__(self, input_size, tree_layers, single_hidden_size, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.single_hidden_size = single_hidden_size
        self.tree_layers = tree_layers
        self.output_size = single_hidden_size * 1
        self.layered_recurrent = LSTMCellDropout(
            input_size=self.input_size,
            hidden_size=self.single_hidden_size,
            dropout=dropout
        )

    def repackage_hidden(self, layered_hidden, forget_vector):
        layered_hidden = forget_hidden_partly_lstm_cell(
            h=layered_hidden,
            forget_vector=forget_vector.unsqueeze(1)
        )  # TODO: check that shit
        return repackage_hidden(layered_hidden)

    def pick_current_output(self, layered_hidden, nodes_depth):
        o_cur = select_layered_hidden(layered_hidden[0], torch.clamp(nodes_depth, min=0, max=self.tree_layers - 1))
        # o_prev = select_layered_hidden(layered_hidden[0], torch.clamp(nodes_depth - 1, min=0, max=self.tree_layers - 1))
        # o_next = select_layered_hidden(layered_hidden[0], torch.clamp(nodes_depth + 1, min=0, max=self.tree_layers - 1))
        return o_cur.squeeze()

    def forward(self, m_input, nodes_depth, layered_hidden, forget_vector, reinit_dropout):
        nodes_depth = torch.clamp(nodes_depth, max=self.tree_layers - 1)
        l_h, l_c = select_layered_lstm_hidden(layered_hidden, nodes_depth)
        l_h, l_c = self.layered_recurrent(m_input, (l_h, l_c), reinit_dropout=reinit_dropout)
        return update_layered_lstm_hidden(layered_hidden, nodes_depth, (l_h, l_c))

    def init_hidden(self, batch_size, cuda, no_grad=False):
        h = setup_tensor(torch.zeros((batch_size, self.tree_layers, self.single_hidden_size)), cuda=cuda)
        c = setup_tensor(torch.zeros((batch_size, self.tree_layers, self.single_hidden_size)), cuda=cuda)

        return h, c


class NT2NLayeredAttentionModel(CombinedModule):
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

        self.num_tree_layers = 50
        self.layered_hidden_size = 100
        self.layered_recurrent = self.module(LayeredRecurrent(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim + self.num_tree_layers,
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
            input_size=self.layered_recurrent.output_size + self.hidden_dim,
            output_size=self.prediction_dim
        ))

    def forward(self, m_input: ASTInput, c_hidden, forget_vector):
        cuda = False
        assert m_input.non_terminals.device == get_device(cuda)
        assert m_input.terminals.device == get_device(cuda)
        assert m_input.nodes_depth.device == get_device(cuda)
        assert c_hidden[0][0].device == get_device(cuda)
        assert c_hidden[0][1].device == get_device(cuda)
        assert c_hidden[1][0].device == get_device(cuda)
        assert c_hidden[1][1].device == get_device(cuda)

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
            node_depths_one_hot = create_one_hot_depths(node_depths[i], node_depths[i].size()[0], self.num_tree_layers)
            layered_hidden = self.layered_recurrent(
                m_input=torch.cat((combined_input[i], node_depths_one_hot), dim=-1),
                nodes_depth=node_depths[i],
                layered_hidden=layered_hidden,
                forget_vector=forget_vector,
                reinit_dropout=reinit_dropout
            )
            current_layered = self.layered_recurrent.pick_current_output(layered_hidden, node_depths[i])
            layered_output_coefficients = self.attn(current_layered, layered_hidden[0])
            layered_output = calc_attention_combination(layered_output_coefficients, layered_hidden[0])
            recurrent_layered_output.append(layered_output)

            cur_h, cur_c = self.recurrent_core(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
            recurrent_output.append(cur_h)

        recurrent_output = torch.stack(recurrent_output, dim=0)
        recurrent_layered_output = torch.stack(recurrent_layered_output, dim=0)
        prediction = self.h2o(torch.cat((recurrent_output, recurrent_layered_output), dim=-1))

        assert hidden is not None
        return prediction, (hidden, layered_hidden)

    def init_hidden(self, batch_size, cuda, no_grad=False):
        return self.recurrent_core.init_hidden(batch_size, cuda), \
               self.layered_recurrent.init_hidden(batch_size, cuda)


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
    layered_hidden = set_layered_hidden(layered_hidden, node_depths, updated)

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
