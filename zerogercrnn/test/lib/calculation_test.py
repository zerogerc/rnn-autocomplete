import torch

from zerogercrnn.lib.calculation import shift_left, pad_tensor, calc_attention_combination, drop_matrix_rows_3d, \
    select_layered_hidden, set_layered_hidden
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


def test_pad_tensor_long():
    tensor = torch.tensor([1, 2, 3], dtype=torch.long)
    assert_tensors_equal(pad_tensor(tensor, seq_len=5), torch.tensor([1, 2, 3, 3, 3], dtype=torch.long))


def test_pad_tensor_float():
    tensor = torch.tensor([0., 0.5, 1.2], dtype=torch.float32)
    assert_tensors_equal(
        pad_tensor(tensor, seq_len=6),
        torch.tensor([0., 0.5, 1.2, 1.2, 1.2, 1.2], dtype=torch.float32)
    )


def test_pad_tensor_2d():
    tensor = torch.tensor([[3, 2, 1], [0, 10, 5]], dtype=torch.long)
    assert_tensors_equal(
        pad_tensor(tensor, seq_len=4),
        torch.tensor(
            [[3, 2, 1], [0, 10, 5], [0, 10, 5], [0, 10, 5]],
            dtype=torch.float32)
    )


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


def test_select_layered_hidden():
    batch_size = 5
    layers = 50
    hidden_size = 10

    node_depths = torch.LongTensor([0, 2, layers - 1, 2, 5])
    layered_hidden = torch.randn((batch_size, layers, hidden_size))

    selected = select_layered_hidden(layered_hidden, node_depths)

    for i in range(node_depths.size()[0]):
        assert torch.nonzero(selected[i][0] == layered_hidden[i][node_depths[i]]).size()[0] == hidden_size


def test_set_layered_hidden():
    batch_size = 6
    layers = 50
    hidden_size = 10

    layered_hidden = torch.randn((batch_size, layers, hidden_size))
    node_depths = torch.LongTensor([0, 1, layers - 1, 2, 5, 1])
    updated = torch.randn((batch_size, hidden_size))
    old_hidden = layered_hidden.clone()

    layered_hidden = set_layered_hidden(layered_hidden, node_depths, updated)

    assert torch.nonzero(old_hidden - layered_hidden).size()[0] == batch_size * hidden_size
    for i in range(node_depths.size()[0]):
        assert torch.nonzero(layered_hidden[i][node_depths[i]] == updated[i]).size()[0] == hidden_size
