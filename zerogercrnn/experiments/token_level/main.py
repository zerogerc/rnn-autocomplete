import argparse

import torch

from zerogercrnn.experiments.common import Main
from zerogercrnn.experiments.token_level.base.main import TokenBaseMain
from zerogercrnn.lib.argutils import add_general_arguments, add_batching_data_args, add_optimization_args, \
    add_recurrent_core_args, add_tokens_args
from zerogercrnn.lib.log import logger

parser = argparse.ArgumentParser(description='Token level neural network')
add_general_arguments(parser)
add_batching_data_args(parser)
add_optimization_args(parser)
add_recurrent_core_args(parser)
add_tokens_args(parser)

parser.add_argument('--prediction', type=str, help='One of: nt2n, nt2n_pre, nt2n_tail, nt2n_sum, nt2nt, ntn2t')
parser.add_argument('--save_model_every', type=int, help='How often to save model', default=1)

# This is for evaluation purposes
parser.add_argument('--eval', action='store_true', help='Evaluate or train')
parser.add_argument('--eval_results_directory', type=str, help='Where to save results of evaluation')


def get_main(args) -> Main:
    if args.prediction == 'token_base':
        main = TokenBaseMain(args)
    else:
        raise Exception('Unknown type of prediction: {}'.format(args.prediciton))

    return main


def train(args):
    get_main(args).train(args)


def evaluate(args):
    get_main(args).eval(args)


if __name__ == '__main__':
    print(torch.__version__)
    _args = parser.parse_args()
    assert _args.title is not None
    logger.should_log = _args.log

    if _args.eval:
        evaluate(_args)
    else:
        train(_args)
