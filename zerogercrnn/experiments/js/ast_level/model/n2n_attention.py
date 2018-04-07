import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.experiments.js.ast_level.model.core import RecurrentCore
from zerogercrnn.experiments.js.ast_level.model.utils import init_layers_uniform
from zerogercrnn.lib.utils.time import logger

"""Base rnn model for JS AST prediction (in use since 01Apr till 01Apr)"""


class NTTailAttentionModel(nn.Module):
    """Model that predicts next non-teminals by the sequence of non-terminals."""

    def __init__(
            self,
            non_terminal_vocab_size,
            embedding_size,
            recurrent_layer: RecurrentCore,
    ):
        super(NTTailAttentionModel, self).__init__()
        assert recurrent_layer.input_size == embedding_size

        self.recurrent_out_size = recurrent_layer.hidden_size
        self.embedding_size = embedding_size
        self.non_terminal_output_size = non_terminal_vocab_size

        # PyTorch is not able to have one optimizer for sparse and non-sparse layers, so we should split parameters
        self.dense_params = []
        self.sparse_params = []

        # Layer that encodes one-hot vector of non-terminals (A)
        self.non_terminal_embedding = self.sparse_model(nn.Embedding(
            num_embeddings=non_terminal_vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        ))

        # Recurrent layer that will have A as an input
        self.recurrent = self.dense_model(
            recurrent_layer
        )

        self.attn = self.dense_model(nn.Linear(
            in_features=self.recurrent_out_size,
            out_features=self.recurrent_out_size,
            bias=False
        ))

        self.cntx_sigmoid = nn.Sigmoid()

        # Layer that transforms hidden state of recurrent layer into next non-terminal
        self.h2NT = self.dense_model(
            nn.Linear(2 * self.recurrent_out_size, self.non_terminal_output_size)
        )
        self.softmaxNT = nn.LogSoftmax(dim=1)

        self._init_params_()

    def forward(self, non_terminal_input, hidden):
        """
        :param non_terminal_input: tensor of size [seq_len, batch_size, 1]
        :param hidden: hidden state of recurrent layer
        """
        seq_len = non_terminal_input.size()[0]
        batch_size = non_terminal_input.size()[1]

        logger.reset_time()
        non_terminal_input = torch.squeeze(non_terminal_input, dim=2)

        # this tensors will be the size of [batch_size, seq_len, embedding_dim]
        non_terminal_emb = self.non_terminal_embedding(non_terminal_input.permute(1, 0))

        non_terminal_emb = non_terminal_emb.permute(1, 0, 2)

        recurrent_input = non_terminal_emb
        logger.log_time_ms('PRE_IN')

        # output_tensor will be the size of (seq_len, batch_size, hidden_size * num_directions)
        recurrent_output, hidden = self.recurrent(recurrent_input, hidden)
        logger.log_time_ms('RECURRENT')

        recurrent_besides_last = recurrent_output.narrow(0, 0, recurrent_output.size()[0] - 1)
        recurrent_last = recurrent_output[-1]
        attn_weights = self.attn(recurrent_besides_last).matmul(recurrent_last[-1])

        # calc cntx vector as sum of h-s multiplied by alpha, then sigmoid
        cntx = attn_weights.permute(1, 0).unsqueeze(1).matmul(recurrent_besides_last.permute(1, 0, 2)).squeeze(1)
        cntx = self.cntx_sigmoid(cntx)

        # concatenate cntx and hidden from last timestamp
        recurrent_attention_output = torch.cat([cntx, recurrent_last], dim=1)

        # converting to pair of (N, T)
        non_terminal_output = self.h2NT(recurrent_attention_output)
        non_terminal_output = self.softmaxNT(non_terminal_output)

        logger.log_time_ms('PRE_OUT')

        return non_terminal_output, hidden

    def sparse_model(self, model):
        self.sparse_params += model.parameters()
        return model

    def dense_model(self, model):
        self.dense_params += model.parameters()
        return model

    def _init_params_(self):
        init_layers_uniform(
            min_value=-0.05,
            max_value=0.05,
            layers=[
                self.non_terminal_embedding,
                self.attn,
                self.h2NT
            ]
        )

    def init_hidden(self, batch_size, cuda):
        return self.recurrent.init_hidden(batch_size, cuda)


class NTTailAttentionModel2Softmax(nn.Module):
    """Model that predicts next non-teminals by the sequence of non-terminals."""

    def __init__(
            self,
            non_terminal_vocab_size,
            embedding_size,
            recurrent_layer: RecurrentCore,
    ):
        super(NTTailAttentionModel2Softmax, self).__init__()
        assert recurrent_layer.input_size == embedding_size

        self.recurrent_out_size = recurrent_layer.hidden_size
        self.embedding_size = embedding_size
        self.non_terminal_output_size = non_terminal_vocab_size

        # PyTorch is not able to have one optimizer for sparse and non-sparse layers, so we should split parameters
        self.dense_params = []
        self.sparse_params = []

        # Layer that encodes one-hot vector of non-terminals (A)
        self.non_terminal_embedding = self.sparse_model(nn.Embedding(
            num_embeddings=non_terminal_vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        ))

        # Recurrent layer that will have A as an input
        self.recurrent = self.dense_model(
            recurrent_layer
        )

        self.attn = self.dense_model(nn.Linear(
            in_features=self.recurrent_out_size,
            out_features=self.recurrent_out_size,
            bias=False
        ))

        self.attn_softmax = nn.Softmax(dim=0)

        # Layer that transforms hidden state of recurrent layer into next non-terminal
        self.h2o = self.dense_model(
            nn.Linear(2 * self.recurrent_out_size, self.non_terminal_output_size)
        )
        self.softmax_o = nn.LogSoftmax(dim=1)

        self._init_params_()

    def forward(self, non_terminal_input, hidden):
        """
        :param non_terminal_input: tensor of size [seq_len, batch_size, 1]
        :param hidden: hidden state of recurrent layer
        """
        seq_len = non_terminal_input.size()[0]
        batch_size = non_terminal_input.size()[1]

        logger.reset_time()
        non_terminal_input = torch.squeeze(non_terminal_input, dim=2)

        # this tensors will be the size of [batch_size, seq_len, embedding_dim]
        non_terminal_emb = self.non_terminal_embedding(non_terminal_input.permute(1, 0))

        non_terminal_emb = non_terminal_emb.permute(1, 0, 2)

        recurrent_input = non_terminal_emb
        logger.log_time_ms('PRE_IN')

        # output_tensor will be the size of (seq_len, batch_size, hidden_size * num_directions)
        recurrent_output, hidden = self.recurrent(recurrent_input, hidden)
        logger.log_time_ms('RECURRENT')

        recurrent_besides_last = recurrent_output.narrow(0, 0, recurrent_output.size()[0] - 1)
        recurrent_last = recurrent_output[-1]
        attn_weights = self.attn(recurrent_besides_last).matmul(recurrent_last[-1])
        attn_weights = self.attn_softmax(attn_weights)

        # calc cntx vector as sum of h-s multiplied by alpha, then sigmoid
        cntx = attn_weights.permute(1, 0).unsqueeze(1).matmul(recurrent_besides_last.permute(1, 0, 2)).squeeze(1)

        # concatenate cntx and hidden from last timestamp
        recurrent_attention_output = torch.cat([cntx, recurrent_last], dim=1)

        o = self.h2o(recurrent_attention_output)
        o = self.softmax_o(o)

        logger.log_time_ms('PRE_OUT')

        return o, hidden

    def sparse_model(self, model):
        self.sparse_params += model.parameters()
        return model

    def dense_model(self, model):
        self.dense_params += model.parameters()
        return model

    def _init_params_(self):
        init_layers_uniform(
            min_value=-0.05,
            max_value=0.05,
            layers=[
                self.non_terminal_embedding,
                self.attn,
                self.h2o
            ]
        )

    def init_hidden(self, batch_size, cuda):
        return self.recurrent.init_hidden(batch_size, cuda)


if __name__ == '__main__':
    core = RecurrentCore(
        input_size=100,
        hidden_size=1500,
        num_layers=1
    )

    model = NTTailAttentionModel2Softmax(
        non_terminal_vocab_size=1000,
        embedding_size=100,
        recurrent_layer=core
    )

    in_tensor = Variable(torch.LongTensor(50, 80, 1).zero_())
    hidden_tensor = Variable(torch.randn((1, 80, 1500)))

    model.forward(non_terminal_input=in_tensor, hidden=hidden_tensor)

    # o_t = torch.randn((50, 80, 1500))
    # w_attn = torch.randn((1500, 1500))
    # h_k = torch.randn(80, 1500)
    #
    # # o_t = o_t.view(-1, 1500)
    # attn_weights = o_t.matmul(w_attn)
    # attn_weights = attn_weights.matmul(h_k[-1])
    #
    # print(torch.bmm(o_t, h_k.expand).size())
