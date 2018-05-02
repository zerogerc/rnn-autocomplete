from abc import abstractmethod

import torch

from zerogercrnn.lib.constants import EMPTY_TOKEN_ID, UNKNOWN_TOKEN_ID


class Metrics:

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


class LossMetrics(Metrics):

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
    """Accuracy metrics that count number of elements different in prediction and target."""

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
        value = float(self.hits) / (self.hits + self.misses)

        if should_print:
            print('Current accuracy: {}'.format(value))

        return value


class MaxPredictionAccuracyMetrics(BaseAccuracyMetrics):

    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def report(self, prediction_target):
        prediction, target = prediction_target
        _, predicted = torch.max(prediction, dim=self.dim)
        super().report((predicted, target))


class TerminalIndexedAccuracyMetrics(Metrics):
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


class TerminalAccuracyMetrics(Metrics):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.general_accuracy = BaseAccuracyMetrics()
        self.empty_accuracy = TerminalIndexedAccuracyMetrics(
            label='Accuracy on terminals that ground truth is <empty>'
        )
        self.non_empty_accuracy = TerminalIndexedAccuracyMetrics(
            label='Accuracy on terminals that ground truth is not <empty>'
        )
        self.ground_not_unk_accuracy = TerminalIndexedAccuracyMetrics(
            label='Accuracy on terminals that ground truth is not <unk> (and ground truth is not <empty>)'
        )
        self.model_not_unk_accuracy = TerminalIndexedAccuracyMetrics(
            label='Accuracy on terminals that model predicted to non <unk> (and ground truth is not <empty>)'
        )

    def drop_state(self):
        self.general_accuracy.drop_state()
        self.empty_accuracy.drop_state()
        self.non_empty_accuracy.drop_state()
        self.ground_not_unk_accuracy.drop_state()
        self.model_not_unk_accuracy.drop_state()

    def report(self, prediction_target):
        prediction, target = prediction_target
        _, predicted = torch.max(prediction, dim=self.dim)
        predicted = predicted.view(-1)
        target = target.view(-1)

        self.general_accuracy.report((predicted, target))

        if not self.is_train:
            empty_indexes = torch.nonzero(target == 0).squeeze()
            self.empty_accuracy.report(predicted, target, empty_indexes)

            non_empty_indexes = torch.nonzero(target - EMPTY_TOKEN_ID).squeeze()
            self.non_empty_accuracy.report(predicted, target, non_empty_indexes)

            predicted = torch.index_select(predicted, 0, non_empty_indexes)
            target = torch.index_select(target, 0, non_empty_indexes)

            ground_not_unk_indexes = torch.nonzero(target - UNKNOWN_TOKEN_ID).squeeze()
            self.ground_not_unk_accuracy.report(predicted, target, ground_not_unk_indexes)

            model_not_unk_indexes = torch.nonzero(predicted - UNKNOWN_TOKEN_ID).squeeze()
            self.model_not_unk_accuracy.report(predicted, target, model_not_unk_indexes)

    def get_current_value(self, should_print=False):
        general_accuracy = self.general_accuracy.get_current_value(should_print=should_print)
        if (not self.is_train) and should_print:
            self.empty_accuracy.get_current_value(should_print=True)
            self.non_empty_accuracy.get_current_value(should_print=True)
            self.ground_not_unk_accuracy.get_current_value(should_print=True)
            self.model_not_unk_accuracy.get_current_value(should_print=True)
        return general_accuracy


class NonTerminalTerminalAccuracyMetrics(Metrics):

    def __init__(self):
        super().__init__()
        self.nt_accuracy = MaxPredictionAccuracyMetrics()
        self.t_accuracy = MaxPredictionAccuracyMetrics()

    def drop_state(self):
        self.nt_accuracy.drop_state()
        self.t_accuracy.drop_state()

    def report(self, data):
        nt_prediction, t_prediction, nt_target, t_target = data
        self.nt_accuracy.report((nt_prediction, nt_target))
        self.t_accuracy.report((t_prediction, t_target))

    def get_current_value(self, should_print=False):
        nt_value = self.nt_accuracy.get_current_value(should_print=False)
        t_value = self.t_accuracy.get_current_value(should_print=False)

        if should_print:
            print('Non terminals accuracy: {}'.format(nt_value))
            print('Terminals accuracy: {}'.format(t_value))

        return nt_value, t_value


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
        for m in self.metrics:
            cur = m.get_current_value(should_print=should_print)
            if result is None:
                result = cur

        return result


if __name__ == '__main__':
    _tensor = torch.LongTensor([[1, 2, 3], [2, 1, 1]]).view(-1)
    _indexes = torch.nonzero(_tensor - 1)
    torch.index_select(_tensor, 0, _indexes.squeeze())
