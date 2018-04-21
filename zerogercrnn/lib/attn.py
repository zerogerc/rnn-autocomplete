import torch.nn as nn
import torch.nn.functional as F

from zerogercrnn.lib.utils import init_layers_uniform


class CyclicBuffer:
    def __init__(self, buffer):
        self.buffer = buffer
        self.it = 0

    def add_vector(self, vector):
        self.buffer[:, self.it, :].copy_(vector)  # TODO: general way
        self.it += 1
        if self.it >= self.buffer.size()[1]:
            self.it = 0

    def get(self):
        return self.buffer


class LastKBuffer:
    def __init__(self, window_len, buffer):
        assert window_len <= buffer.size()[1]
        self.buffer_size = buffer.size()[1]
        self.window_len = window_len
        self.buffer = buffer

        self.it = window_len

    def add_vector(self, vector):
        self.buffer[:, self.it, :].copy_(vector)  # TODO: general way
        self.it += 1
        if self.it >= self.buffer_size:
            self.buffer.narrow(dimension=1, start=0, length=self.window_len).copy_(
                self.buffer.narrow(dimension=1, start=self.buffer_size - self.window_len, length=self.window_len)
            )
            self.it = self.window_len

    def get(self):
        return self.buffer.narrow(dimension=1, start=self.it - self.window_len, length=self.window_len)


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            init_layers_uniform(-0.05, 0.05, [self.attn])

        # elif self.method == 'concat':
        #     self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        #     self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))
        #     nn.init.uniform(self.attn.parameters(), -0.05, 0.05)
        #     nn.init.uniform(self.other, -0.05, 0.05)

    def parameters(self):
        return super().parameters()

    def sparse_parameters(self):
        return []

    def forward(self, main_vector, attn_vectors):
        """
        :param main_vector: matrix of size [batch_size, N]
        :param attn_vectors: matrix of size [batch_size, seq_len, N]
        :return:
        """
        seq_len = attn_vectors.size()[1]

        # Calculate energies for each encoder output
        attn_energies = self.score(main_vector, attn_vectors)

        return F.softmax(attn_energies, dim=1)

    def score(self, main_vector, attn_vectors):
        """
        :param main_vector: matrix of size [batch_size, N]
        :param attn_vectors: matrix of size [batch_size, seq_len, N]
        :return: matrix with attention coefficients of size [batch_size, seq_len, 1]
        """
        if self.method == 'dot':
            pass  # all is ready
        elif self.method == 'general':
            attn_vectors = self.attn(attn_vectors)
        else:
            raise Exception('Unknown attention method: {}'.format(self.method))

        # main_vector [batch_size, N] -> [batch_size, 1, 1, N]
        main_vector = main_vector.unsqueeze(1).unsqueeze(1)
        # att_vectors [batch_size, seq_len, N, 1]
        attn_vectors = attn_vectors.unsqueeze(3)
        # after multiplication [batch_size, seq_len, 1, 1] -> [batch_size, seq_len, 1, 1]
        energy = main_vector.matmul(attn_vectors).squeeze(-1)
        return energy

        # TODO: implement concat
        # elif self.method == 'concat':
        #     energy = self.attn(torch.cat((hidden, encoder_output), 1))
        #     energy = self.other.dot(energy)
        #     return energy
