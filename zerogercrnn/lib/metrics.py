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
    def get_current_value(self):
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

    def get_current_value(self):
        if isinstance(self.total_loss, Variable):
            loss_value = self.total_loss.data[0]
        else:
            loss_value = self.total_loss

        return loss_value / self.total_count


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

    def get_current_value(self):
        return float(self.hits) / (self.hits + self.misses)
