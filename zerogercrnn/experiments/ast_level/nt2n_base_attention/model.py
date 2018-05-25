import torch
import numpy as np

from zerogercrnn.experiments.ast_level.data import ASTInput
from zerogercrnn.lib.core import CombinedModule, EmbeddingsModule, RecurrentCore, LinearLayer, LSTMCellDropout
from zerogercrnn.lib.attn import Attn
from zerogercrnn.lib.utils import repackage_hidden, forget_hidden_partly, get_best_device, setup_tensor, forget_hidden_partly_lstm_cell
from zerogercrnn.lib.calculation import calc_attention_combination, set_layered_hidden


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
        # self.buffer = self.buffer.mul(forget_vector.unsqueeze(1)) TODO: implement forgetting
        self.buffer = [repackage_hidden(b) for b in self.buffer]


class LastKAttention(CombinedModule):
    def __init__(self, hidden_size, k=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.context_buffer = None
        self.attn = self.module(Attn(method='general', hidden_size=self.hidden_size))

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

        self.context_buffer.add_vector(current_hidden)
        return attn_output


class NT2NBaseAttentionModel(CombinedModule):
    """Base Model with attention on last n hidden states of LSTM."""

    def __init__(
            self,
            non_terminals_num,
            non_terminal_embedding_dim,
            terminals_num,
            terminal_embedding_dim,
            hidden_dim,
            num_layers,
            dropout
    ):
        super().__init__()

        self.non_terminals_num = non_terminals_num
        self.non_terminal_embedding_dim = non_terminal_embedding_dim
        self.terminals_num = terminals_num
        self.terminal_embedding_dim = terminal_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

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

        # self.recurrent_core = self.module(RecurrentCore(
        #     input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
        #     hidden_size=self.hidden_dim,
        #     num_layers=self.num_layers,
        #     dropout=self.dropout,
        #     model_type='lstm'
        # ))

        self.recurrent_cell = self.module(LSTMCellDropout(
            input_size=self.non_terminal_embedding_dim + self.terminal_embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout
        ))

        self.last_k_attention = self.module(LastKAttention(
            hidden_size=self.hidden_dim,
            k=50
        ))

        self.h2o = self.module(LinearLayer(
            input_size=2 * self.hidden_dim,
            output_size=self.non_terminals_num
        ))

    def forward(self, m_input: ASTInput, hidden, forget_vector):
        non_terminal_input = m_input.non_terminals
        terminal_input = m_input.terminals

        nt_embedded = self.nt_embedding(non_terminal_input)
        t_embedded = self.t_embedding(terminal_input)

        combined_input = torch.cat([nt_embedded, t_embedded], dim=2)

        recurrent_output, new_hidden, attn_output = self.get_recurrent_layers_outputs(
            combined_input=combined_input,
            hidden=hidden,
            forget_vector=forget_vector
        )

        concatenated_output = torch.cat((recurrent_output, attn_output), dim=-1)
        prediction = self.h2o(concatenated_output)

        return prediction, new_hidden

    def get_recurrent_layers_outputs(
            self, combined_input, hidden, forget_vector):
        hidden = repackage_hidden(forget_hidden_partly_lstm_cell(hidden, forget_vector=forget_vector))
        self.last_k_attention.repackage_and_forget_buffer_partly(forget_vector)

        recurrent_output = []
        layered_attn_output = []
        for i in range(combined_input.size()[0]):
            reinit_dropout = i == 0

            # core recurrent part
            cur_h, cur_c = self.recurrent_cell(combined_input[i], hidden, reinit_dropout=reinit_dropout)
            hidden = (cur_h, cur_c)
            recurrent_output.append(cur_h)

            # layered part
            attn_output = self.last_k_attention(cur_h)
            layered_attn_output.append(attn_output)


        # combine outputs from different layers
        recurrent_output = torch.stack(recurrent_output, dim=0)
        layered_attn_output = torch.stack(layered_attn_output, dim=0)

        return recurrent_output, hidden, layered_attn_output

    def init_hidden(self, batch_size):
        self.last_k_attention.init_hidden(batch_size)
        return self.recurrent_cell.init_hidden(batch_size)
