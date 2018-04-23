import argparse

import torch

from zerogercrnn.experiments.ast_level.nt2n.main import NT2NMain
from zerogercrnn.experiments.ast_level.nt2n_sum.main import NT2NSumAttentionMain
from zerogercrnn.experiments.ast_level.nt2n_tail.main import NT2NTailAttentionMain
from zerogercrnn.experiments.ast_level.nt2nt.main import NT2NTMain
from zerogercrnn.experiments.ast_level.ntn2t.main import NTN2TMain
from zerogercrnn.lib.argutils import add_general_arguments, add_batching_data_args, add_optimization_args, \
    add_recurrent_core_args, add_non_terminal_args, add_terminal_args
from zerogercrnn.lib.log import logger

parser = argparse.ArgumentParser(description='AST level neural network')
add_general_arguments(parser)
add_batching_data_args(parser)
add_optimization_args(parser)
add_recurrent_core_args(parser)
add_non_terminal_args(parser)
add_terminal_args(parser)
parser.add_argument('--terminal_embeddings_file', type=str, help='File with pretrained terminal embeddings')
parser.add_argument('--prediction', type=str, help='One of: nt2n, nt2n_tail, nt2n_sum, nt2nt, ntn2t')
parser.add_argument('--eval', action='store_true', help='Evaluate or train')


def get_main(args):
    if args.prediction == 'nt2n':
        main = NT2NMain(args)
    elif args.prediction == 'nt2n_tail':
        main = NT2NTailAttentionMain(args)
    elif args.prediction == 'nt2n_sum':
        main = NT2NSumAttentionMain(args)
    elif args.prediction == 'nt2nt':
        main = NT2NTMain(args)
    elif args.prediction == 'ntn2t':
        main = NTN2TMain(args)
    else:
        raise Exception('Not supported prediction type: {}'.format(args.prediction))

    return main


def train(args):
    get_main(args).train(args)


def evaluate(args):
    if args.saved_model is None:
        print('WARNING: Running eval without saved_model. Not a good idea')
    get_main(args).eval(args, print_every=1)


if __name__ == '__main__':
    _args = parser.parse_args()
    assert _args.title is not None
    logger.should_log = _args.log

    if _args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if not _args.cuda:
        print("WARNING: You are running without cuda. Is it ok?")

    if _args.eval:
        evaluate(_args)
    else:
        train(_args)
