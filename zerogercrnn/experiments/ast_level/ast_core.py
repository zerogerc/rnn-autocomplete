from abc import abstractmethod
import torch
from zerogercrnn.lib.utils import setup_tensor, repackage_hidden
from zerogercrnn.lib.core import CombinedModule, AlphaBetaSumLayer, EmbeddingsModule, LinearLayer
from zerogercrnn.lib.attn import Attn
from zerogercrnn.lib.calculation import calc_attention_combination

from zerogercrnn.experiments.ast_level.data import ASTInput

class ASTNT2NModule(CombinedModule):

    def __init__(
            self,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminals_num,
            terminal_embedding_dim,
            recurrent_output_size
    ):
        super().__init__()

        self.non_terminals_num = non_terminals_num
        self.non_terminal_embedding_dim = non_terminal_embedding_dim
        self.terminals_num = terminals_num
        self.terminal_embedding_dim = terminal_embedding_dim
        self.recurrent_output_size = recurrent_output_size

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

        self.h2o = self.module(LinearLayer(
            input_size=self.recurrent_output_size,
            output_size=self.non_terminals_num
        ))


    @abstractmethod
    def get_recurrent_output(self, combined_input, ast_input: ASTInput, m_hidden, forget_vector):
        pass

    def forward(self, ast_input: ASTInput, m_hidden, forget_vector):
        non_terminal_input = ast_input.non_terminals
        terminal_input = ast_input.terminals

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)
        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        recurrent_output, new_m_hidden = self.get_recurrent_output(
            combined_input=combined_input,
            ast_input=ast_input,
            m_hidden=m_hidden,
            forget_vector=forget_vector
        )

        m_output = self.h2o(recurrent_output)
        return m_output, m_hidden


class LastKBuffer:
    def __init__(self, window_len, hidden_size):
        self.buffer = None
        self.window_len = window_len
        self.hidden_size = hidden_size
        self.it = 0

    def add_vector(self, vector):
        self.buffer[self.it] = vector
        self.it += 1
        if self.it >= self.window_len:
            self.it = 0

    def get(self):
        return torch.stack(self.buffer, dim=1)


def init_buffer(self, batch_size):
    self.buffer = [setup_tensor(torch.zeros((batch_size, self.hidden_size))) for _ in range(self.window_len)]


def repackage_and_forget_buffer_partly(self, forget_vector):
    self.buffer = [repackage_hidden(b.mul(forget_vector)) for b in self.buffer]


class LastKAttention(CombinedModule):
    def __init__(self, hidden_size, k=50, ab_transform=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.ab_transform = ab_transform

        self.context_buffer = None
        self.attn = self.module(Attn(method='general', hidden_size=self.hidden_size))
        if self.ab_transform:
            self.alpha_beta_sum = self.module(AlphaBetaSumLayer(min_value=-1, max_value=2))

    def repackage_and_forget_buffer_partly(self, forget_vector):
        self.context_buffer.repackage_and_forget_buffer_partly(forget_vector)

    def init_hidden(self, batch_size):
        self.context_buffer = LastKBuffer(window_len=self.k, hidden_size=self.hidden_size)
        self.context_buffer.init_buffer(batch_size)

    def forward(self, current_hidden):
        if self.context_buffer is None:
            raise Exception('You should init buffer first')

        current_buffer = self.context_buffer.get()
        attn_output_coefficients = self.attn(current_hidden, current_buffer)
        attn_output = calc_attention_combination(attn_output_coefficients, current_buffer)

        buffer_vector = current_hidden
        if self.ab_transform:
            buffer_vector = self.alpha_beta_sum(current_hidden, attn_output)

        self.context_buffer.add_vector(buffer_vector)
        return attn_output


class LastKAttentionBase(CombinedModule):
    def __init__(self, hidden_size, k=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k

        self.context_buffer = None
        self.attn = self.module(Attn(method='general', hidden_size=self.hidden_size))

    def add_vector(self, vector):
        self.context_buffer.add_vector(vector)

    def forward(self, current_hidden):
        if self.context_buffer is None:
            raise Exception('You should init buffer first')

        current_buffer = self.context_buffer.get()
        attn_output_coefficients = self.attn(current_hidden, current_buffer)
        attn_output = calc_attention_combination(attn_output_coefficients, current_buffer)

        return attn_output

    def repackage_and_forget_buffer_partly(self, forget_vector):
        self.context_buffer.repackage_and_forget_buffer_partly(forget_vector)

    def init_hidden(self, batch_size):
        self.context_buffer = LastKBuffer(window_len=self.k, hidden_size=self.hidden_size)
        self.context_buffer.init_buffer(batch_size)
