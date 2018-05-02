import torch


def shift_left(matrix, dimension):
    """Shift tensor left by one along specified dimension. This operation performed in-place"""
    m_len = matrix.size()[dimension]
    matrix.narrow(dim=dimension, start=0, length=m_len - 1) \
        .copy_(matrix.narrow(dim=dimension, start=1, length=m_len - 1))


def calc_attention_combination(attention_weights, matrix):
    """Calculate sum of vectors of matrix along dim=1 with coefficients specified by attention_weights.

    :param attention_weights: size - [batch_size, seq_len, 1]
    :param matrix: size - [batch_size, seq_len, vector_dim]
    :return: matrix of size [batch_size, vector_dim]
    """
    return attention_weights.transpose(1, 2).matmul(matrix).squeeze(1)


def drop_matrix_rows_3d(matrix, forget_vector):
    """
    Zeroing blocks along first dimension according to forget_vector. Forget vector should consist of 0s and 1s.

    :param matrix: size - [N1, N2, N3]
    :param forget_vector: size - [N1, 1]
    :return: size - [N1, N2, N3]
    """
    return matrix.mul(forget_vector.unsqueeze(2))


def select_layered_hidden(layered_hidden, node_depths):
    """Selects hidden state for each element in the batch according to layer number in node_depths

    :param layered_hidden: tensor of size [batch_size, layers_num, hidden_size]
    :param node_depths: for each batch line contains layer that should be picked. shape: [batch_size]
    """
    batch_size = layered_hidden.size()[0]
    layers_num = layered_hidden.size()[1]
    hidden_size = layered_hidden.size()[2]
    depths_one_hot = layered_hidden.new(batch_size, layers_num)

    depths_one_hot.zero_().scatter_(1, node_depths.unsqueeze(1), 1)
    mask = depths_one_hot.unsqueeze(2).byte()
    mask = mask.to(layered_hidden.device)

    return torch.masked_select(layered_hidden, mask).view(batch_size, 1, hidden_size)


def set_layered_hidden(layered_hidden, node_depths, updated):
    """Returns new tensor that represents updated hidden state. Only layers that specified in node_depths get updated.

    :param layered_hidden: tensor of size [batch_size, layers_num, hidden_size]
    :param node_depths: for each batch line contains layer that should be updated. shape: [batch_size]
    :param updated: updated hidden states for particular layer. shape: [batch_size, hidden_size]
    """
    batch_size = layered_hidden.size()[0]
    layers_num = layered_hidden.size()[1]
    hidden_size = layered_hidden.size()[2]

    node_depths_update = node_depths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, hidden_size)
    updated = updated.unsqueeze(1)
    node_depths_update.to(layered_hidden.device)

    return layered_hidden.scatter(1, node_depths_update, updated)
