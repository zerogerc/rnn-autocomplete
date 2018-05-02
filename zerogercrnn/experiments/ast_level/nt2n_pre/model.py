import torch

from zerogercrnn.lib.core import PretrainedEmbeddingsModule, RecurrentCore, \
    LinearLayer, CombinedModule
from zerogercrnn.lib.embedding import Embeddings
from zerogercrnn.lib.utils import forget_hidden_partly, repackage_hidden


class NT2NBothTNTPretrainedModel(CombinedModule):
    def __init__(
            self,
            non_terminal_embeddings: Embeddings,
            terminal_embeddings: Embeddings,
            hidden_dim,
            prediction_dim,
            num_layers,
            dropout
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.prediction_dim = prediction_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.nt_embedding = self.module(PretrainedEmbeddingsModule(
            embeddings=non_terminal_embeddings,
            requires_grad=False,
            sparse=False
        ))
        self.non_terminal_embedding_dim = self.nt_embedding.embedding_dim

        self.t_embedding = self.module(PretrainedEmbeddingsModule(
            embeddings=terminal_embeddings,
            requires_grad=False,
            sparse=False
        ))
        self.terminal_embedding_dim = self.t_embedding.embedding_dim

        self.recurrent_core = self.module(RecurrentCore(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type='lstm'
        ))

        self.h2o = self.module(LinearLayer(
            input_size=self.hidden_dim,
            output_size=self.prediction_dim,
        ))

    def forward(self, non_terminal_input, terminal_input, hidden, forget_vector):
        assert non_terminal_input.size() == terminal_input.size()
        assert non_terminal_input.size() == terminal_input.size()

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)

        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        hidden = repackage_hidden(hidden)
        hidden = forget_hidden_partly(hidden, forget_vector=forget_vector)
        recurrent_output, new_hidden = self.recurrent_core(combined_input, hidden)

        prediction = self.h2o(recurrent_output)

        return prediction, new_hidden

    def init_hidden(self, batch_size, cuda, no_grad=False):
        return self.recurrent_core.init_hidden(batch_size, cuda, no_grad=no_grad)
