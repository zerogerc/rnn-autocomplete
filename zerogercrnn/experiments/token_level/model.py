import torch.nn as nn

from zerogercrnn.experiments.ast_level.model.core import RecurrentCore, LogSoftmaxOutputLayer
from zerogercrnn.experiments.ast_level.model.utils import forget_hidden_partly, repackage_hidden


class TokenLevelBaseModel(nn.Module):

    def __init__(self, embedding_size, tokens_number, hidden_size, num_layers=1, dropout=0.):
        super().__init__()

        self.recurrent = RecurrentCore(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            model_type='lstm'
        )

        self.h2o = LogSoftmaxOutputLayer(
            input_size=hidden_size,
            output_size=tokens_number,
            dim=2
        )

    def forward(self, input, hidden, forget_vector):
        assert hidden is not None
        seq_len, batch_size, emb_size = input.size()

        hidden = repackage_hidden(hidden)
        hidden = forget_hidden_partly(hidden, forget_vector=forget_vector)
        output, hidden = self.recurrent(input, hidden)

        prediction = self.h2o(output)

        return prediction, hidden

    def init_hidden(self, batch_size, cuda, no_grad=False):
        return self.recurrent.init_hidden(batch_size, cuda, no_grad=no_grad)
