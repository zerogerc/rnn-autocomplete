import numpy as np
import torch
import os

from zerogercrnn.experiments.ast_level.data import ASTTarget
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper
from zerogercrnn.experiments.ast_level.metrics import SingleNonTerminalAccuracyMetrics
from zerogercrnn.experiments.token_level.metrics import UnknownIsMissTokenMetrics
from zerogercrnn.lib.metrics import SequentialMetrics, BaseAccuracyMetrics, TopKAccuracy


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


def run_metrics(reader, metrics):
    metrics.drop_state()
    metrics.report((
        torch.from_numpy(reader.predicted),
        torch.from_numpy(reader.target),
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


def calc_top_accuracies(results_dir):
    reader = ResultsReader(results_dir=results_dir)
    metrics = TopKAccuracy(k=5)

    metrics.drop_state()
    metrics.report((
        torch.from_numpy(reader.predicted),
        torch.from_numpy(reader.target)
    ))
    metrics.get_current_value(should_print=True)


def convert_to_top1(dir_from, dir_to):
    reader = ResultsReader(results_dir=dir_from)
    np.save(os.path.join(dir_to, 'predicted'), reader.predicted[:, :, 0])
    np.save(os.path.join(dir_to, 'target'), reader.target)


def token_eval(res_dir, save_dir):
    reader = ResultsReader(results_dir=res_dir)
    metrics = UnknownIsMissTokenMetrics()
    run_metrics(reader, metrics)


def main(task, model):
    common_dirs = {
        'base': 'eval_verified/nt2n_base',
        'attention': 'eval_verified/nt2n_base_attention',
        'layered': 'eval_verified/nt2n_base_attention_plus_layered',
        'token': 'eval_verified/token_base'
    }
    topk_dirs = {
        'base': 'eval_verified/nt2n_base/top5',
        'attention': 'eval_verified/nt2n_base_attention/top5',
        'layered': 'eval_verified/nt2n_base_attention_plus_layered/top5',
        'token': 'eval_verified/token_base/top5'
    }
    save_dir = 'eval_local'

    if task == 'per_nt':
        res_dir = common_dirs[model]
        eval_nt(
            results_dir=res_dir,
            save_dir=save_dir,
            group=True
        )
    elif task == 'token_eval':
        res_dir = common_dirs[model]
        token_eval(res_dir, save_dir)
    elif task == 'topk':
        res_dir = topk_dirs[model]
        calc_top_accuracies(results_dir=res_dir)
    elif task == 'to_top1':
        convert_to_top1(topk_dirs[model], common_dirs[model])
    else:
        raise Exception('Unknown task type')


if __name__ == '__main__':
    _tasks = ['topk', 'per_nt', 'to_top1', 'token_eval']
    # main(task='to_top1', model='token')
    main(task='token_eval', model='token')
