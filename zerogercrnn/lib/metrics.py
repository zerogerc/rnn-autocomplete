from abc import abstractmethod
import torch
from torch.autograd import Variable


class Metrics:

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
        if isinstance(self.total_loss, Variable):
            loss_value = self.total_loss.data[0]
        else:
            loss_value = self.total_loss

        loss_value = loss_value / self.total_count
        if should_print:
            print('Current loss: {}'.format(loss_value))
        return loss_value


class AccuracyMetrics(Metrics):

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.reported = 0

    def drop_state(self):
        self.hits = 0
        self.misses = 0
        self.reported = 0

    def report(self, prediction_target):
        prediction, target = prediction_target
        self.reported += 1

        _, predicted = torch.max(prediction, dim=2)
        current_misses = torch.nonzero(predicted - target).size()[0]
        current_hits = target.view(-1).size()[0] - current_misses

        self.hits += current_hits
        self.misses += current_misses

    def get_current_value(self, should_print=False):
        value = float(self.hits) / (self.hits + self.misses)

        if should_print:
            print('Current loss: {}'.format(value))

        return value


class NonTerminalTerminalAccuracyMetrics(Metrics):

    def __init__(self):
        self.nt_accuracy = AccuracyMetrics()
        self.t_accuracy = AccuracyMetrics()

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
