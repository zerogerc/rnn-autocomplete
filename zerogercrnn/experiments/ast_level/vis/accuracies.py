import numpy as np
import torch

from zerogercrnn.experiments.ast_level.data import ASTTarget
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper
from zerogercrnn.experiments.ast_level.metrics import SingleNonTerminalAccuracyMetrics
from zerogercrnn.lib.metrics import SequentialMetrics, BaseAccuracyMetrics


# region Utils

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


def get_per_nt_result(results_dir, save_dir, group=False):
    reader = ResultsReader(results_dir=results_dir)
    metrics = SingleNonTerminalAccuracyMetrics(
        non_terminals_file='data/ast/non_terminals.json',
        results_dir=save_dir,
        group=group,
        dim=None
    )

    run_nt_metrics(reader, metrics)


# endregion


def eval_nt(results_dir, save_dir, group=False):
    reader = ResultsReader(results_dir=results_dir)

    metrics = SequentialMetrics([
        NonTerminalsMetricsWrapper(BaseAccuracyMetrics()),
        SingleNonTerminalAccuracyMetrics(
            non_terminals_file='data/ast/non_terminals.json',
            results_dir=save_dir,
            group=group,
            dim=None
        )
    ])

    run_nt_metrics(reader, metrics)


def get_res_dir(model_type):
    if model_type == 'nt2n_base':
        return 'eval_verified/nt2n_base_30k'
    elif model_type == 'nt2n_base_attention':
        return 'eval_verified/nt2n_base_attention_30k'
    else:
        raise Exception('Unknown model_type')


def main():
    res_dir = get_res_dir(model_type='nt2n_base')
    save_dir = 'eval_local'

    eval_nt(
        results_dir=res_dir,
        save_dir=save_dir,
        group=False
    )


if __name__ == '__main__':
    main()
