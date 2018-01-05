import torch
import pytest
from mock import Mock

from lib.data.batcher import BatchNode

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
