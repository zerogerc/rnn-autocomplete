import torch

from zerogercrnn.lib.old.converter import convert_row_text_to_index_tensor, convert_row_text_to_one_hot
from zerogercrnn.testutils.utils import assert_numbers_almost_equal, assert_tensors_equal


def test_convert_row_text_to_one_hot():
    row_text = 'ab'
    all_letters = 'abc'

    one_hot = convert_row_text_to_one_hot(row_text, all_letters)

    assert one_hot.size() == torch.Size([2, 3])

    assert_numbers_almost_equal(one_hot[0][0], 1)
    assert_numbers_almost_equal(one_hot[0][1], 0)
    assert_numbers_almost_equal(one_hot[0][2], 0)

    assert_numbers_almost_equal(one_hot[1][0], 0)
    assert_numbers_almost_equal(one_hot[1][1], 1)
    assert_numbers_almost_equal(one_hot[1][2], 0)

    assert_tensors_equal(
        one_hot,
        torch.FloatTensor([[1., 0., 0.], [0., 1., 0.]])
    )


def test_convert_row_text_to_index_tensor():
    row_text = 'abc'
    all_letters = 'bac'

    positions = convert_row_text_to_index_tensor(row_text, all_letters)

    assert_tensors_equal(positions, torch.LongTensor([1, 0, 2]))
