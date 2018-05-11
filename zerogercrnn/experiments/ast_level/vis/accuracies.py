import numpy as np
import torch

from zerogercrnn.lib.metrics import BaseAccuracyMetrics


class ResultsReader:
    """Class that could read results from lib.metrics.ResultsSaver and then produce matrices for visualization."""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.predicted = np.load(self.results_dir + '/predicted')
        self.target = np.load(self.results_dir + '/target')


if __name__ == '__main__':
    reader = ResultsReader(results_dir='data/ast/temp')
    metrics = BaseAccuracyMetrics()

    metrics.drop_state()
    metrics.report((
        torch.from_numpy(reader.predicted),
        torch.from_numpy(reader.target)
    ))
    metrics.get_current_value(should_print=True)
