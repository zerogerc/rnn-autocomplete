import torch
import numpy as np

from zerogercrnn.lib.metrics import TensorVisualizerMetrics, TensorVisualizer2DMetrics, TensorVisualizer3DMetrics
from zerogercrnn.testutils.utils import assert_tensors_equal


def test_tensor_visualizer_metrics():
    metrics = TensorVisualizerMetrics(file=None)

    t1 = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
    t2 = torch.from_numpy(np.array([[3, 4, 5], [6, 7, 8]]))

    metrics.drop_state()
    metrics.report(t1)
    metrics.report(t2)

    expected = (t1 + t2) / 2

    assert_tensors_equal(metrics.get_current_value(), expected)


def test_tensor_visualizer_metrics_random():
    torch.manual_seed(123)
    metrics = TensorVisualizerMetrics(file=None)

    t1 = torch.randn((2, 3))
    t2 = torch.randn((2, 3))

    metrics.drop_state()
    metrics.report(t1)
    metrics.report(t2)

    expected = (t1 + t2) / 2

    assert_tensors_equal(metrics.get_current_value(), expected, eps=1e-6)


def test_tensor_visualizer2d_metrics_dim0():
    metrics = TensorVisualizer2DMetrics(dim=0, file=None)

    t1 = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
    t2 = torch.from_numpy(np.array([[3, 4, 5], [6, 7, 8]]))

    metrics.drop_state()
    metrics.report(t1)
    metrics.report(t2)

    expected = torch.sum(t1 + t2, dim=0).float() / 4

    assert_tensors_equal(metrics.get_current_value(), expected)


def test_tensor_visualizer2d_metrics_dim0_random():
    torch.manual_seed(100)
    metrics = TensorVisualizer2DMetrics(dim=0, file=None)

    t1 = torch.randn((2, 3))
    t2 = torch.randn((2, 3))

    metrics.drop_state()
    metrics.report(t1)
    metrics.report(t2)

    expected = torch.sum(t1 + t2, dim=0).float() / 4

    assert_tensors_equal(metrics.get_current_value(), expected, eps=1e-6)


def test_tensor_visualizer2d_metrics_dim1():
    metrics = TensorVisualizer2DMetrics(dim=1, file=None)

    t1 = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
    t2 = torch.from_numpy(np.array([[3, 4, 5], [6, 7, 8]]))

    metrics.drop_state()
    metrics.report(t1)
    metrics.report(t2)

    expected = torch.sum(t1 + t2, dim=1).float() / 6

    assert_tensors_equal(metrics.get_current_value(), expected)


def test_tensor_visualizer2d_metrics_dim1_random():
    torch.manual_seed(145)
    metrics = TensorVisualizer2DMetrics(dim=1, file=None)

    t1 = torch.randn((2, 3))
    t2 = torch.randn((2, 3))

    metrics.drop_state()
    metrics.report(t1)
    metrics.report(t2)

    expected = torch.sum(t1 + t2, dim=1).float() / 6

    assert_tensors_equal(metrics.get_current_value(), expected, eps=1e-6)


def test_tensor_visualizer3d_metrics():
    torch.manual_seed(2)
    metrics = TensorVisualizer3DMetrics(file=None)

    t1 = torch.from_numpy(np.array([
        [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4],
        ],
        [
            [1, 1, 1, 1, 8],
            [2, 10, 2, 4, 2],
            [3, 3, 3, 3, 3],
            [9, 4, 4, 4, 4],
        ],
        [

            [1, 1, -10, 1, 9],
            [2, -1, 2, 5, 2],
            [0, 3, 7, 3, 3],
            [4, 2, 4, 4, 4],
        ]
    ]))
    t2 = torch.from_numpy(np.array([
        [
            [10, 1, 10, 1, 1],
            [2, 22, 28, 29, 2],
            [3, 34, 33, 3, 31],
            [4, 4, 4, 44, 4],
        ],
        [
            [1, 11, 1, 1, 8],
            [2, 10, 2, 4, 2],
            [3, 3, 33, 3, 3],
            [9, 4, 4, 47, 40],
        ],
        [

            [1000, 1, -10, 1, 91],
            [2, -111, 2, 5, 2],
            [0, 32, 7, 3, 3],
            [4, 24, 4, 4, 4],
        ]
    ]))

    metrics.drop_state()
    metrics.report(t1)
    metrics.report(t2)

    expected = (t1 + t2).sum(0).sum(0) / (12 * 2)

    assert_tensors_equal(metrics.get_current_value(), expected, eps=1e-6)


def test_tensor_visualizer3d_metrics_random():
    torch.manual_seed(2)
    metrics = TensorVisualizer3DMetrics(file=None)

    t1 = torch.randn((3, 4, 5))
    t2 = torch.randn((3, 4, 5))

    metrics.drop_state()
    metrics.report(t1)
    metrics.report(t2)

    expected = (t1 + t2).sum(0).sum(0) / (12 * 2)

    assert_tensors_equal(metrics.get_current_value(), expected, eps=1e-6)
