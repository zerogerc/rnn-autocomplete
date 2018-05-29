from zerogercrnn.lib.metrics import Metrics, BaseAccuracyMetrics, IndexedAccuracyMetrics


class AggregatedTokenMetrics(Metrics):

    def __init__(self):
        super().__init__()
        self.common = BaseAccuracyMetrics()
        self.target_non_unk = IndexedAccuracyMetrics('Target not unk')
        self.prediction_non_unk = IndexedAccuracyMetrics('Prediction not unk')

    def drop_state(self):
        self.common.drop_state()
        self.target_non_unk.drop_state()
        self.prediction_non_unk.drop_state()

    def report(self, prediction_target):
        prediction, target = prediction_target
        prediction = prediction.view(-1)
        target = target.view(-1)

        self.common.report((prediction, target))

        pred_non_unk_indices = (prediction != 0).nonzero().squeeze()
        target_non_unk_indices = (target != 0).nonzero().squeeze()

        self.prediction_non_unk.report(prediction, target, pred_non_unk_indices)
        self.target_non_unk.report(prediction, target, target_non_unk_indices)

    def get_current_value(self, should_print=False):
        print('P1 = {}'.format(self.common.get_current_value(False)))
        print('P2 = {}'.format(self.prediction_non_unk.metrics.hits / (self.common.hits + self.common.misses)))
        print('P3 = {}'.format(self.target_non_unk.get_current_value(False)))
        print('P4 = {}'.format(self.prediction_non_unk.get_current_value(False)))
