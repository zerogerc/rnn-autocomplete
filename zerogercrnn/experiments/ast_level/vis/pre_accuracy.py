import torch
import numpy as np
import json
import os
from zerogercrnn.lib.accuracies import indexed_topk_hits, topk_hits
from zerogercrnn.experiments.ast_level.data import ASTTarget
from zerogercrnn.lib.metrics import Metrics
from zerogercrnn.lib.preprocess import read_json
from zerogercrnn.lib.constants import EOF_TOKEN

JS_NON_TERMINALS = 'data/ast/non_terminals.json'

# region Utils

class ResultsReader:
    """Class that could read results from lib.metrics.ResultsSaver and then produce matrices for visualization."""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.predicted = torch.from_numpy(np.load(self.results_dir + '/predicted'))
        self.target = torch.from_numpy(np.load(self.results_dir + '/target'))

    def get_nt_predicted_target(self):
        return self.predicted, ASTTarget(self.target, None)


def run_nt_metrics(reader: ResultsReader, metrics: Metrics):
    metrics.drop_state()
    metrics.report(reader.get_nt_predicted_target())
    metrics.get_current_value(should_print=True)


# endregion Utils

def get_per_nt_hits_and_misses(nt_id, predicted, target):
    index = (target == nt_id).nonzero().squeeze()
    return indexed_topk_hits(predicted, target, index)


def transform_to_per_nt_topk(reader, non_terminals):
    assert torch.max(reader.target) <= len(non_terminals) - 1

    res = {}

    predicted = reader.predicted.view(-1, reader.predicted.size()[-1])
    target = reader.target.view(-1)

    for id, nt in enumerate(non_terminals):
        topk_hits, total = get_per_nt_hits_and_misses(id, predicted, target)

        for i in range(topk_hits.size(0)):
            key = 'top' + str(i)
            if key not in res:
                res[key] = {}

            if nt not in res[key]:
                res[key][nt] = {}
                res[key][nt]['hits'] = 0
                res[key][nt]['misses'] = 0

            res[key][nt]['hits'] += topk_hits[i].item()
            res[key][nt]['misses'] += total - topk_hits[i].item()

    return res


def task_transform_to_per_nt_topk(results_dir, non_terminals_file, res_dir):
    reader = ResultsReader(results_dir=results_dir)
    non_terminals = read_json(non_terminals_file)
    non_terminals.append(EOF_TOKEN)

    res = transform_to_per_nt_topk(reader, non_terminals)
    with open(os.path.join(res_dir, 'topk.json'), mode='w') as f:
        f.write(json.dumps(res))


def main():
    task_transform_to_per_nt_topk(
        results_dir='eval_verified/nt2n_base_large_embeddings_30k',
        non_terminals_file=JS_NON_TERMINALS,
        res_dir='eval_local'
    )
    # tasks = ['per_nt_top_k']


if __name__ == '__main__':
    main()
