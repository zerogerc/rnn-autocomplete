import torch.nn as nn


class GRULinuxNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, dropout=0.):
        super(GRULinuxNetwork, self).__init__()
        self.input_size = 1
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size)
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.init_params()

    def init_params(self):
        for name, param in self.embedding.named_parameters():
            nn.init.uniform(param, -0.08, 0.08)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

        for name, param in self.h2o.named_parameters():
            nn.init.uniform(param, -0.08, 0.08)

    def forward(self, input_tensor):
        """
        :param input_tensor: tensor of size [seq_len, batch_size, 1]
        """
        seq_len = input_tensor.size()[0]
        batch_size = input_tensor.size()[1]

        emb = self.embedding(input_tensor.view(seq_len, batch_size))
        output_tensor, hidden = self.gru(emb)

        o_sz = list(output_tensor.size())
        o_sz[-1] = self.output_size
        output_tensor = output_tensor.view(-1, self.hidden_size)
        output_tensor = self.h2o(output_tensor)
        output_tensor = self.softmax(output_tensor)
        output_tensor = output_tensor.view(o_sz)

        return output_tensor
