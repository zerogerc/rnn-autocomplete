import json
import os

from zerogercrnn.lib.constants import EOF_TOKEN
from zerogercrnn.lib.preprocess import read_json

JS_NON_TERMINALS = 'data/ast/non_terminals.json'


# region Utils

def save_pretty_json(json_file):
    jsn = read_json(json_file)
    with open(os.path.splitext(json_file)[0] + '_pretty.json', mode='w') as f:
        f.write(json.dumps(jsn, indent=4, sort_keys=True))


def accuracy(hits, misses):
    if hits + misses == 0:
        return 0
    return float(hits) / (hits + misses)


# endregion


class NtMapUtils:
    @staticmethod
    def per_nt_accuracies(mp):
        res = {}
        for key, value in mp.items():
            res[key] = accuracy(hits=value['hits'], misses=value['misses'])
        return res

    @staticmethod
    def total_accuracy(mp):
        hits = 0
        misses = 0
        for key, value in mp.items():
            hits += value['hits']
            misses += value['misses']
        return accuracy(hits=hits, misses=misses)

    @staticmethod
    def grouped_per_nt_accuracies(mp):
        hits = {}
        misses = {}
        for key, value in mp.items():
            if key != EOF_TOKEN:
                key = key[:-2]

            hits[key] = hits.get(key, 0) + value['hits']
            misses[key] = misses.get(key, 0) + value['misses']

        res = {}
        for k in hits.keys():
            res[k] = accuracy(hits=hits[k], misses=misses[k])
        return res


def draw_top_1(file):
    res = read_json(os.path.join(file, 'topk.json'))
    print(NtMapUtils.total_accuracy(res['top0']))
    print(json.dumps(
        NtMapUtils.grouped_per_nt_accuracies(res['top0']),
        indent=4,
        sort_keys=True
    ))


def main():
    res_dir_base_large_embeddings = 'eval_verified/nt2n_base_large_embeddings_30k'
    draw_top_1(res_dir_base_large_embeddings)


if __name__ == '__main__':
    save_pretty_json('eval_verified/nt2n_base_large_embeddings_30k/topk.json')
    # main()
