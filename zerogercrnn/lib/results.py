import torch


class AccuracyMeasurer:
    """Accuray measurer for models that have one int as target."""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.reported = 0

    def add_predictions(self, prediction, target):
        self.reported += 1

        current_misses = torch.nonzero(prediction - target).size()[0]
        current_hits = target.view(-1).size()[0] - current_misses

        self.hits += current_hits
        self.misses += current_misses

    def get_current_accuracy(self):
        return float(self.hits) / (self.hits + self.misses)
