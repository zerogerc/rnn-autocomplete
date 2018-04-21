def shift_left(matrix, dimension):
    """Shift tensor left by one along specified dimension. This operation performed in-place"""
    m_len = matrix.size()[dimension]
    matrix.narrow(dimension=dimension, start=0, length=m_len - 1) \
        .copy_(matrix.narrow(dimension=dimension, start=1, length=m_len - 1))


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
    return matrix.mul(forget_vector.unsqueeze(2), out=matrix)
