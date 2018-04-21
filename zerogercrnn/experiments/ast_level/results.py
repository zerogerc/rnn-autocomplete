import argparse

import torch

from zerogercrnn.lib.log import tqdm_lim
from zerogercrnn.lib.argutils import add_general_arguments, add_batching_data_args, add_optimization_args, \
    add_recurrent_core_args, add_non_terminal_args, add_terminal_args
from zerogercrnn.experiments.ast_level.nt2n.main import NT2NMain
from zerogercrnn.experiments.ast_level.nt2nt.main import NT2NTMain
from zerogercrnn.experiments.ast_level.ntn2t.main import NTN2TMain
from zerogercrnn.lib.log import logger
from zerogercrnn.lib.metrics import AccuracyMetrics

parser = argparse.ArgumentParser(description='AST level neural network')
add_general_arguments(parser)
add_batching_data_args(parser)
add_optimization_args(parser)
add_recurrent_core_args(parser)
add_non_terminal_args(parser)
add_terminal_args(parser)
parser.add_argument('--terminal_embeddings_file', type=str, help='File with pretrained terminal embeddings')
parser.add_argument('--prediction', type=str, help='One of: nt2n, nt2nt, ntn2t')


def get_main(args):
    if args.prediction == 'nt2n':
        main = NT2NMain(args)
    elif args.prediction == 'nt2nt':
        main = NT2NTMain(args)
    elif args.prediction == 'ntn2t':
        main = NTN2TMain(args)
    else:
        raise Exception('Not supported prediction type: {}'.format(args.prediction))

    return main


def print_results(args):
    assert args.prediction == 'nt2n'

    # seed = 1000
    # random.seed(seed)
    # numpy.random.seed(seed)

    main = get_main(args)

    routine = main.validation_routine

    metrics = AccuracyMetrics()
    metrics.drop_state()
    main.model.eval()

    for iter_num, iter_data in enumerate(tqdm_lim(main.data_generator.get_eval_generator(), lim=1000)):
        metrics_data = routine.run(iter_num, iter_data)
        metrics.report(metrics_data)
        metrics.get_current_value(should_print=True)


if __name__ == '__main__':
    _args = parser.parse_args()
    assert _args.title is not None
    logger.should_log = _args.log

    if _args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if not _args.cuda:
        print("WARNING: You are running without cuda. Is it ok?")

    print_results(_args)
