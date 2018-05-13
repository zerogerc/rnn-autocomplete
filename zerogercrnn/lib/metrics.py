import os
from abc import abstractmethod

import numpy as np
import torch


class Metrics:
    """Base class for all metrics. Metrics is runned on results of model run either during training or evaluation.
    It can return some value to use it for plotting, print some current metrics to console or save results to file."""

    def __init__(self):
        self.is_train = True

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    @abstractmethod
    def drop_state(self):
        pass

    @abstractmethod
    def report(self, *values):
        pass

    @abstractmethod
    def get_current_value(self, should_print=False):
        pass

    def decrease_hits(self, number):
        """Used to drop hits in the appended tail of train data."""
        print('Decrease hits not implemented!!!')


class MetricsCombination(Metrics):
    """Combination of metrics to perform only lightweight checks during training and a full analysis during eval."""

    def __init__(self, train_metrics: Metrics, eval_metrics: Metrics):
        super().__init__()
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics

    def drop_state(self):
        if self.is_train:
            self.train_metrics.drop_state()
        else:
            self.eval_metrics.drop_state()

    def report(self, *values):
        if self.is_train:
            self.train_metrics.report(*values)
        else:
            self.eval_metrics.report(*values)

    def get_current_value(self, should_print=False):
        if self.is_train:
            return self.train_metrics.get_current_value(should_print)
        else:
            return self.eval_metrics.get_current_value(should_print)

    def decrease_hits(self, number):
        if self.is_train:
            self.train_metrics.decrease_hits(number)
        else:
            self.eval_metrics.decrease_hits(number)


class LossMetrics(Metrics):
    """Metric that calculates average loss."""

    def __init__(self):
        super().__init__()
        self.total_loss = None
        self.total_count = 0

    def drop_state(self):
        self.total_loss = None
        self.total_count = 0

    def report(self, loss):
        if self.total_loss is None:
            self.total_loss = loss
        else:
            self.total_loss += loss

        self.total_count += 1

    def get_current_value(self, should_print=False):
        loss_value = self.total_loss.item() / self.total_count
        if should_print:
            print('Current loss: {}'.format(loss_value))
        return loss_value


class BaseAccuracyMetrics(Metrics):
    """Metrics that count accuracy as a number of different elements between prediction and target."""

    def __init__(self):
        super().__init__()
        self.hits = 0
        self.misses = 0
        self.reported = 0

    def decrease_hits(self, number):
        print('Hits decreased by {}'.format(number))
        self.hits -= number

    def drop_state(self):
        self.hits = 0
        self.misses = 0
        self.reported = 0

    def report(self, prediction_target):
        prediction, target = prediction_target
        self.reported += 1

        current_misses = torch.nonzero(prediction - target)
        if len(current_misses.size()) == 0:
            current_misses = 0
        else:
            current_misses = current_misses.size()[0]
        current_hits = target.view(-1).size()[0] - current_misses

        self.hits += current_hits
        self.misses += current_misses

    def get_current_value(self, should_print=False):
        if self.hits + self.misses == 0:
            return 0
        value = float(self.hits) / (self.hits + self.misses)

        if should_print:
            print('Current accuracy: {}'.format(value))

        return value


class MaxPredictionWrapper(Metrics):
    """Metrics that should be used as wrapper. During report calculate max on prediction along specified dimension
    and pass this to the base metrics."""

    def __init__(self, base: Metrics, dim=2):
        super().__init__()
        self.base = base
        self.dim = dim

    def drop_state(self):
        self.base.drop_state()

    def report(self, prediction_target):
        prediction, target = prediction_target
        _, predicted = torch.max(prediction, dim=self.dim)
        self.base.report((predicted, target))

    def get_current_value(self, should_print=False):
        return self.base.get_current_value(should_print=should_print)

    def decrease_hits(self, number):
        self.base.decrease_hits(number)


class MaxPredictionAccuracyMetrics(BaseAccuracyMetrics):
    """Metrics that calculates max along specified dimension on prediction and then pass result to BaseAccuracyMetrics.
    """

    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def report(self, prediction_target):
        prediction, target = prediction_target
        _, predicted = torch.max(prediction, dim=self.dim)
        super().report((predicted, target))


class IndexedAccuracyMetrics(Metrics):
    """Metrics that calculates accuracy, but only on elements from index. Other elements are ignored.
    Basic building block for specific metrics, like accuracy per non-terminal."""

    def __init__(self, label):
        super().__init__()
        self.label = label
        self.metrics = BaseAccuracyMetrics()

    def drop_state(self):
        self.metrics.drop_state()

    def report(self, predicted, target, indexes):
        predicted = torch.index_select(predicted, 0, indexes)
        target = torch.index_select(target, 0, indexes)
        self.metrics.report((predicted, target))

    def get_current_value(self, should_print=False):
        value = self.metrics.get_current_value(should_print=False)

        if should_print:
            print('{}: {}'.format(self.label, value))

        return value

    def decrease_hits(self, number):
        self.metrics.decrease_hits(number)


class ResultsSaver(Metrics):
    """Metrics that perform saving of predicted and target tensors to file."""

    def __init__(self, dir_to_save):
        super().__init__()
        self.file_to_save = dir_to_save
        self.predicted = []
        self.target = []

    def drop_state(self):
        self.predicted = []
        self.target = []

    def report(self, predicted_target):
        predicted, target = predicted_target
        self.predicted.append(predicted.cpu().numpy())
        self.target.append(target.cpu().numpy())

    def get_current_value(self, should_print=False):
        """Saves value to file."""
        predicted = np.concatenate(self.predicted, axis=0)
        target = np.concatenate(self.target, axis=0)

        if not os.path.exists(self.file_to_save):
            os.makedirs(self.file_to_save)
        predicted.dump(self.file_to_save + '/predicted')
        target.dump(self.file_to_save + '/target')


class SequentialMetrics(Metrics):
    """Metrics that reports results to a sequence of metrics.

    get_current_value will return value of first metrics.
    So all metrics would print their output during report but only value of first will be used by caller.
    """

    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def drop_state(self):
        for m in self.metrics:
            m.drop_state()

    def report(self, prediction_target):
        for m in self.metrics:
            m.report(prediction_target)

    def get_current_value(self, should_print=False):
        result = None
        print('-------------------------------------------')
        for m in self.metrics:
            cur = m.get_current_value(should_print=should_print)
            print('-------------------------------------------')
            if result is None:
                result = cur

        return result

    def decrease_hits(self, number):
        for m in self.metrics:
            m.decrease_hits(number)


if __name__ == '__main__':
    _tensor = torch.LongTensor([[1, 2, 3], [2, 1, 1]]).view(-1)
    _indexes = torch.nonzero(_tensor - 1)
    torch.index_select(_tensor, 0, _indexes.squeeze())
