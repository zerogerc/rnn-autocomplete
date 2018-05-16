import json
import os

import torch
import numpy as np

from zerogercrnn.experiments.ast_level.utils import read_non_terminals
from zerogercrnn.lib.constants import EMPTY_TOKEN_ID, UNKNOWN_TOKEN_ID, EOF_TOKEN
from zerogercrnn.lib.metrics import Metrics, BaseAccuracyMetrics, IndexedAccuracyMetrics, MaxPredictionAccuracyMetrics
from zerogercrnn.lib.utils import get_best_device


class NonTerminalsMetricsWrapper(Metrics):
    """Metrics that extract non-terminals from target and pass non-terminals tensor to base metrics."""

    def __init__(self, base: Metrics):
        super().__init__()
        self.base = base

    def drop_state(self):
        self.base.drop_state()

    def report(self, prediction_target):
        prediction, target = prediction_target
        self.base.report((prediction, target.non_terminals))

    def get_current_value(self, should_print=False):
        return self.base.get_current_value(should_print)

    def decrease_hits(self, number):
        self.base.decrease_hits(number)


class SingleNonTerminalAccuracyMetrics(Metrics):
    """Metrics that show accuracies per non-terminal. It should not be used for plotting, but to
    print results on console during model evaluation."""

    def __init__(self, non_terminals_file, results_dir=None, group=False, dim=2):
        """

        :param non_terminals_file: file with json of non-terminals
        :param results_dir: where to save json with accuracies per non-terminal
        :param dim: dimension to run max function on for predicted values
        """
        super().__init__()
        print('SingleNonTerminalAccuracyMetrics created!')

        self.non_terminals = read_non_terminals(non_terminals_file)
        self.non_terminals_number = len(self.non_terminals)
        self.results_dir = results_dir
        self.group = group
        self.dim = dim

        self.accuracies = [IndexedAccuracyMetrics(label='ERROR') for _ in self.non_terminals]

    def drop_state(self):
        for accuracy in self.accuracies:
            accuracy.drop_state()

    def report(self, data):
        prediction, target = data
        if self.dim is None:
            predicted = prediction
        else:
            _, predicted = torch.max(prediction, dim=self.dim)
        predicted = predicted.view(-1)
        target = target.non_terminals.view(-1)

        for cur in range(len(self.non_terminals)):
            indices = (target == cur).nonzero().squeeze()
            self.accuracies[cur].report(predicted, target, indices)

    def get_current_value(self, should_print=False):
        result = []
        for cur in range(len(self.non_terminals)):
            cur_accuracy = self.accuracies[cur].get_current_value(should_print=False)
            result.append(cur_accuracy)
            if should_print:
                print('Accuracy on {} is {}'.format(self.non_terminals[cur], cur_accuracy))

        self.save_to_file(result)

        return 0  # this metrics if only for printing

    def save_to_file(self, result):
        if self.results_dir is not None:
            if self.group:
                nt, res = self.get_grouped_result()
            else:
                nt, res = self.non_terminals, result

            with open(os.path.join(self.results_dir, 'nt_acc.txt'), mode='w') as f:
                f.write(json.dumps(nt))
                f.write('\n')
                f.write(json.dumps(res))

    def get_grouped_result(self):
        """Calc accuracies ignoring last two bits of information."""
        nt = set()
        hits = {}
        misses = {}
        for i in range(len(self.non_terminals)):
            base = self.non_terminals[i]
            if self.non_terminals[i] != EOF_TOKEN:
                base = base[:-2]  # remove last two bits

            nt.add(base)

            if base not in hits:
                hits[base] = 0
            if base not in misses:
                misses[base] = 0

            hits[base] += self.accuracies[i].metrics.hits
            misses[base] += self.accuracies[i].metrics.misses

        nt = sorted(list(nt))
        result = []

        nt.remove('Program')
        nt.remove('AssignmentPattern')
        for cur in nt:
            if hits[cur] + misses[cur] == 0:
                result.append(0)
            else:
                result.append(float(hits[cur]) / (hits[cur] + misses[cur]))

        return nt, result


class TerminalAccuracyMetrics(Metrics):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.general_accuracy = BaseAccuracyMetrics()
        self.empty_accuracy = IndexedAccuracyMetrics(
            label='Accuracy on terminals that ground truth is <empty>'
        )
        self.non_empty_accuracy = IndexedAccuracyMetrics(
            label='Accuracy on terminals that ground truth is not <empty>'
        )
        self.ground_not_unk_accuracy = IndexedAccuracyMetrics(
            label='Accuracy on terminals that ground truth is not <unk> (and ground truth is not <empty>)'
        )
        self.model_not_unk_accuracy = IndexedAccuracyMetrics(
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


class LayeredNodeDepthsAttentionMetrics(Metrics):
    """Metrics that is able to visualize attention coefficient per node depths"""

    def __init__(self):
        super().__init__()
        self.per_depth_attention_sum = np.zeros((50, 50))

    def drop_state(self):
        pass

    def report(self, node_depths, attention_coefficients):
        for i in range(50):
            index = torch.nonzero((node_depths == i))
            if index.size()[0] == 0:
                continue
            selected_attention = torch.index_select(attention_coefficients, dim=0, index=index.squeeze())
            selected_attention = selected_attention.squeeze()
            to_report = torch.sum(selected_attention, dim=0).numpy()
            self.per_depth_attention_sum[i] += to_report / index.size()[0]

    def get_current_value(self, should_print=False):
        np.save('eval_local/attention/per_depth_matrix', self.per_depth_attention_sum)
        return 0  # this metrics is only for saving results to file.



class ConcatenatedAttentionMetrics(Metrics):
    """Metrics that is able to visualize attention coefficient per node depths"""

    def __init__(self):
        super().__init__()
        self.sum = torch.from_numpy(np.zeros(2000)).float().to(get_best_device())
        self.reported = 0

    def drop_state(self):
        pass

    def report(self, output):
        self.sum += torch.sum(torch.sum(output, dim=0), dim=0) / (output.size()[0] * output.size()[1])
        self.reported += 1

    def get_current_value(self, should_print=False):
        np.save('eval/temp/output_sum_matrix', self.sum.detach().cpu().numpy())
        return 0  # this metrics is only for saving results to file.
