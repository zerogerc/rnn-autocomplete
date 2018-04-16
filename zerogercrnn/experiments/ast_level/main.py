import argparse

import torch

from zerogercrnn.experiments.argutils import add_general_arguments, add_batching_data_args, add_optimization_args, \
    add_recurrent_core_args, add_non_terminal_args, add_terminal_args
from zerogercrnn.experiments.ast_level.nt2n.main import NT2NMain
from zerogercrnn.experiments.ast_level.nt2nt.main import NT2NTMain
from zerogercrnn.lib.utils.time import logger

parser = argparse.ArgumentParser(description='AST level neural network')
add_general_arguments(parser)
add_batching_data_args(parser)
add_optimization_args(parser)
add_recurrent_core_args(parser)
add_non_terminal_args(parser)
add_terminal_args(parser)
parser.add_argument('--terminal_embeddings_file', type=str, help='File with pretrained terminal embeddings')
parser.add_argument('--prediction', type=str, help='One of: nt2n, nt2nt')


def train(args):
    if args.prediction == 'nt2n':
        main = NT2NMain(args)
    elif args.prediction == 'nt2nt':
        main = NT2NTMain(args)
    else:
        raise Exception('Not supported prediction type: {}'.format(args.prediction))

    main.run(args)


if __name__ == '__main__':
    _args = parser.parse_args()
    assert _args.title is not None
    logger.should_log = _args.log

    if _args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    train(_args)
