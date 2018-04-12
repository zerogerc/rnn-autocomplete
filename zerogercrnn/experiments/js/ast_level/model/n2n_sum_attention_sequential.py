import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.experiments.js.ast_level.model.core import LSTMCellDropout, LogSoftmaxOutputLayer, \
    ContextBasedSumAttention
from zerogercrnn.experiments.js.ast_level.model.utils import init_layers_uniform, forget_hidden_partly
from zerogercrnn.lib.utils.time import logger


class NTSumlAttentionModelSequential(nn.Module):
    """Model that predicts next non-teminals by the sequence of non-terminals.
    """

    def __init__(
            self,
            non_terminal_vocab_size,
            embedding_size,
            hidden_size,
            seq_len
    ):
        super(NTSumlAttentionModelSequential, self).__init__()

        self.train = True

        self.non_terminal_vocab_size = non_terminal_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # PyTorch is not able to have one optimizer for sparse and non-sparse layers, so we should split parameters
        self.dense_params = []
        self.sparse_params = []

        # Layer that encodes one-hot vector of non-terminals (A)
        self.non_terminal_embedding = self.dense_model(
            nn.Embedding(
                num_embeddings=self.non_terminal_vocab_size,
                embedding_dim=self.embedding_size,
                sparse=False
            )
        )

        # Recurrent layer that will have (A) as an input
        self.lstm_cell = self.dense_model(
            LSTMCellDropout(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                dropout=0.25
            )
        )

        # Layer that applies attention to past self.cntx hidden states of contexts
        self.sum_attention = self.dense_model(
            ContextBasedSumAttention(
                seq_len=self.seq_len,
                hidden_size=self.hidden_size
            )
        )

        # Layer that transforms context output layer into next non-terminal
        self.h2o = self.dense_model(
            LogSoftmaxOutputLayer(
                input_size=self.hidden_size,
                output_size=self.non_terminal_vocab_size,
                dim=1  # TODO: check that sht
            )
        )

        self._init_params_()

    def forget_context_partly(self, forget_vector):
        self.sum_attention.forget_context_partly(forget_vector)

    def forward(self, non_terminal_input, hidden, forget_vector, reinit_dropout=False):
        """
        :param non_terminal_input: tensor of size [seq_len, batch_size, 1]
        :param hidden: hidden state of recurrent layer
        :param forget_vector: vector to drop context if program finishes.
                              format: [batch_size, 1] filled with either 0 or 1
        :type reinit_dropout whether to reinit dropout mask of LSTMCell
        """
        logger.reset_time()

        non_terminal_input = torch.squeeze(non_terminal_input, dim=1)
        non_terminal_emb = self.non_terminal_embedding(non_terminal_input)

        logger.log_time_ms('EMBEDDING')

        recurrent_hidden_state, recurrent_cell_state = self.lstm_cell(
            input_tensor=non_terminal_emb,
            hidden_state=hidden,
            apply_dropout=self.train,
            reinit_dropout=reinit_dropout
        )

        logger.log_time_ms('RECURRENT')

        # drop context for programs that finished
        # self.sum_attention.forget_context_partly(forget_vector)

        attention_hidden = self.sum_attention(recurrent_hidden_state)
        logger.log_time_ms('ATTENTION')

        prediction = self.h2o(attention_hidden)

        logger.log_time_ms('OUTPUT')

        # attention_hidden instead of recurrent_hidden here is main idea of network.
        # We will propagate only needed information to the next layer.
        return prediction, (attention_hidden, recurrent_cell_state)

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
            ]
        )

    def init_hidden(self, batch_size, cuda):
        return self.lstm_cell.init_hidden(batch_size, cuda)


if __name__ == '__main__':
    model = NTSumlAttentionModelSequential(
        non_terminal_vocab_size=1000,
        embedding_size=100,
        hidden_size=100,
        seq_len=5
    )

    batch_size = 80

    in_tensor = Variable(torch.LongTensor(batch_size, 1).zero_())
    hidden_tensor = model.init_hidden(batch_size, cuda=False)
    forget_vector = torch.ones((batch_size, 1))

    for i in range(10):
        in_tensor = Variable(in_tensor.data.fill_(i))
        if i > 0:
            forget_vector[0][0] = 0
        prediction, hidden_tensor = model.forward(
            non_terminal_input=in_tensor,
            hidden=hidden_tensor,
            forget_vector=forget_vector,
            reinit_dropout=True
        )
