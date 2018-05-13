import numpy as np
import torch

from zerogercrnn.experiments.ast_level.data import ASTTarget
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper
from zerogercrnn.experiments.ast_level.metrics import SingleNonTerminalAccuracyMetrics
from zerogercrnn.lib.metrics import SequentialMetrics, BaseAccuracyMetrics


class ResultsReader:
    """Class that could read results from lib.metrics.ResultsSaver and then produce matrices for visualization."""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.predicted = np.load(self.results_dir + '/predicted')
        self.target = np.load(self.results_dir + '/target')


def run_nt_metrics(reader, metrics):
    metrics.drop_state()
    metrics.report((
        torch.from_numpy(reader.predicted),
        ASTTarget(torch.from_numpy(reader.target), None)
    ))
    metrics.get_current_value(should_print=True)


def get_accuracy_result(results_dir):
    reader = ResultsReader(results_dir=results_dir)
    metrics = NonTerminalsMetricsWrapper(BaseAccuracyMetrics())
    run_nt_metrics(reader, metrics)


def get_per_nt_result(results_dir, save_dir):
    reader = ResultsReader(results_dir=results_dir)
    metrics = SingleNonTerminalAccuracyMetrics(
        non_terminals_file='data/ast/non_terminals.json',
        results_dir=save_dir,
        group=True,
        dim=None
    )

    run_nt_metrics(reader, metrics)


def eval_nt(results_dir):
    reader = ResultsReader(results_dir=results_dir)

    metrics = SequentialMetrics([
        NonTerminalsMetricsWrapper(BaseAccuracyMetrics()),
        SingleNonTerminalAccuracyMetrics(
            non_terminals_file='data/ast/non_terminals.json',
            results_dir=None,
            group=True,
            dim=None
        )
    ])

    run_nt_metrics(reader, metrics)


if __name__ == '__main__':
    base_res_dir = 'eval/ast/nt2n_base'
    base_save_dir = 'eval_local'
    layered_attention_res_dir = 'eval/ast/nt2n_layered_attention'
    layered_attention_save_dir = 'eval_local'

    get_per_nt_result(
        results_dir=base_res_dir,
        save_dir=base_save_dir
    )

    # eval_nt(results_dir=base_res_dir)
