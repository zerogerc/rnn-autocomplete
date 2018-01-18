import pytest
import torch
from mock import Mock

from zerogerc_rnn.lib.old.batcher import BatchNode

TEST_LEN = 1001
TEST_BATCH = 10


@pytest.fixture
def mock_input_picker():
    return Mock(return_value=torch.LongTensor())


@pytest.fixture
def mock_output_picker():
    return Mock(return_value=torch.LongTensor())


class TestBatchNode:
    def test_get_batched_random_should_work(self, mock_input_picker, mock_output_picker):
        node = BatchNode('test', TEST_LEN, mock_input_picker, mock_output_picker)
        next(node.get_batched_random(TEST_BATCH))

        mock_input_picker.assert_called()
        mock_output_picker.assert_called()

        # all unique
        input_ids = set(mock_input_picker.call_args[0][0])
        target_ids = set(mock_output_picker.call_args[0][0])
        assert input_ids == target_ids
        assert len(input_ids) == TEST_BATCH
        assert len(target_ids) == TEST_BATCH

        next(node.get_batched_random(TEST_BATCH))
        assert set(mock_input_picker.call_args[0][0]) != input_ids

    def test_get_batched_epoch_should_work(self, mock_input_picker, mock_output_picker):
        node = BatchNode('test', TEST_LEN, mock_input_picker, mock_output_picker)

        input_ids = []
        target_ids = []
        for _ in node.get_batched_epoch(TEST_BATCH):
            input_ids += set(mock_input_picker.call_args[0][0])
            target_ids += set(mock_output_picker.call_args[0][0])

        assert mock_input_picker.call_count == (TEST_LEN // TEST_BATCH)
        assert mock_output_picker.call_count == (TEST_LEN // TEST_BATCH)

        assert set(input_ids) == set(target_ids)
        assert len(set(input_ids)) == TEST_LEN - (TEST_LEN % TEST_BATCH)
        assert len(set(target_ids)) == TEST_LEN - (TEST_LEN % TEST_BATCH)


# class TestDataPickers:
#     def test_general_data_picker_text_to_index(self):
#         all_letters = 'xyz'
#         data = general_data_picker_new(
#             'xxyyzz',
#             2,
#             [0, 3],
#             lambda row: convert_row_text_to_index_tensor(row, all_letters)
#         )
#
#         assert data.size() == torch.Size([2, 2, 1])
#         assert_tensors_equal(data[:, 0, :], torch.LongTensor([[0], [0]]))
#         assert_tensors_equal(data[:, 1, :], torch.LongTensor([[1], [2]]))
