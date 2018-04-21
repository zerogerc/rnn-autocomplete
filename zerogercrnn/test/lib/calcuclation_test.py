import torch
from zerogercrnn.lib.calculation import shift_left, calc_attention_combination, drop_matrix_rows_3d
from zerogercrnn.testutils.utils import assert_tensors_equal


def test_move_right_should_move_dim0():
    matrix = torch.LongTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    dimension = 0
    shift_left(matrix, dimension)
    assert_tensors_equal(matrix, torch.LongTensor([[2, 2, 2], [3, 3, 3], [3, 3, 3]]))


def test_move_right_should_move_dim1():
    matrix = torch.LongTensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    dimension = 1
    shift_left(matrix, dimension)
    assert_tensors_equal(matrix, torch.LongTensor([[2, 3, 3], [2, 3, 3], [2, 3, 3]]))


def test_calc_attention_combination_should_work():
    matrix = torch.FloatTensor([
        [
            [1, 10],
            [1, 1],
            [1, 8]
        ],
        [
            [1, 1],
            [4, 4],
            [9, 6]
        ]
    ])

    attention_weights = torch.FloatTensor([
        [
            [1. / 2],
            [1.],
            [1. / 2]
        ],
        [
            [1.],
            [1. / 2],
            [1. / 3]
        ]
    ])

    expected = torch.FloatTensor([
        [2., 10.],
        [6., 5.]
    ])

    attentioned = calc_attention_combination(attention_weights, matrix)
    assert_tensors_equal(attentioned, expected)


def test_drop_matrix_rows_3d():
    matrix = torch.FloatTensor([
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ],
        [
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6]
        ]
    ])

    forget_vector = torch.FloatTensor([
        [0],
        [1]
    ])

    expected = torch.FloatTensor([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6]
        ]
    ])

    assert_tensors_equal(drop_matrix_rows_3d(matrix, forget_vector), expected)
