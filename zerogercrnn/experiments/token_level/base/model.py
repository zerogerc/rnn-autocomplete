from zerogercrnn.experiments.token_level.core import TokenModel
from zerogercrnn.lib.core import RecurrentCore
from zerogercrnn.lib.utils import forget_hidden_partly, repackage_hidden


class TokenBaseModel(TokenModel):
    def __init__(self, num_tokens, embedding_dim, hidden_size):
        super().__init__(num_tokens=num_tokens, embedding_dim=embedding_dim, recurrent_output_size=hidden_size)
        self.hidden_size = hidden_size

        self.lstm = self.module(RecurrentCore(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=0.,
            model_type='lstm'
        ))

    def get_recurrent_output(self, input_embedded, hidden, forget_vector):
        hidden = repackage_hidden(forget_hidden_partly(hidden, forget_vector))
        return self.lstm(input_embedded, hidden)

    def init_hidden(self, batch_size):
        return self.lstm.init_hidden(batch_size)
