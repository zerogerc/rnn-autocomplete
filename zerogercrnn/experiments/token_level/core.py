from abc import abstractmethod

import torch

from zerogercrnn.lib.core import CombinedModule, EmbeddingsModule, LinearLayer


class TokenModelBase(CombinedModule):
    def __init__(self, num_tokens, embedding_dim, recurrent_output_size):
        super().__init__()

        self.token_embeddings = self.module(EmbeddingsModule(
            num_embeddings=num_tokens,
            embedding_dim=embedding_dim,
            sparse=True
        ))

        self.h2o = self.module(LinearLayer(
            input_size=recurrent_output_size,
            output_size=num_tokens,
            bias=True
        ))

    def forward(self, m_input: torch.Tensor, hidden: torch.Tensor, forget_vector: torch.Tensor):
        input_embedded = self.token_embeddings(m_input)
        recurrent_output, hidden = self.get_recurrent_output(input_embedded, hidden, forget_vector)
        m_output = self.h2o(recurrent_output)
        return m_output, hidden

    @abstractmethod
    def get_recurrent_output(self, input_embedded, hidden, forget_vector):
        return None, None
