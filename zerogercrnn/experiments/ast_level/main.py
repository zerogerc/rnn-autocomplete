import argparse
import os

import torch

from zerogercrnn.experiments.ast_level.nt2n_base.main import NT2NBaseMain
from zerogercrnn.experiments.ast_level.nt2n_layered.main import NT2NLayeredMain
from zerogercrnn.experiments.ast_level.nt2n_layered_attention.main import NT2NLayeredAttentionMain
from zerogercrnn.experiments.ast_level.nt2n_layered_prob_attention.main import NT2NLayeredProbabilisticAttentionMain
from zerogercrnn.experiments.ast_level.nt2n_tail.main import NT2NTailAttentionMain
from zerogercrnn.experiments.ast_level.nt2n_te.main import NT2NPretrainedTerminalsMain
from zerogercrnn.experiments.ast_level.ntn2t.main import NTN2TMain
from zerogercrnn.experiments.ast_level.ntn2t_base.main import NTN2TBaseMain
from zerogercrnn.experiments.ast_level.ntn2t_tail.main import NTN2TTailAttentionMain
from zerogercrnn.experiments.ast_level.vis.model import visualize_tensor, draw_line_plot
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

parser.add_argument('--prediction', type=str, help='One of: nt2n, nt2n_pre, nt2n_tail, nt2n_sum, nt2nt, ntn2t')
parser.add_argument('--save_model_every', type=int, help='How often to save model', default=1)

# This is for evaluation purposes
parser.add_argument('--eval', action='store_true', help='Evaluate or train')
parser.add_argument('--eval_results_directory', type=str, help='Where to save results of evaluation')

# Grid search parameters
parser.add_argument('--grid_name', type=str, help='Parameter to grid search')
parser.add_argument(
    '--grid_values', nargs='+', type=int,
    help='Values for grid searching'
)  # how to make it int or float?

# Additional parameters for specific models
parser.add_argument(
    '--nodes_depths_stat_file', type=str,
    help='File with number of times particular depth is occurred in train file'
)


def get_main(args):
    if args.prediction == 'nt2n_base':
        main = NT2NBaseMain(args)
    elif args.prediction == 'nt2n_te':
        main = NT2NPretrainedTerminalsMain(args)
    elif args.prediction == 'nt2n_layered':
        main = NT2NLayeredMain(args)
    elif args.prediction == 'nt2n_layered_attention':
        main = NT2NLayeredAttentionMain(args)
    elif args.prediction == 'nt2n_layered_prob_attention':
        main = NT2NLayeredProbabilisticAttentionMain(args)
    elif args.prediction == 'ntn2t_base':
        main = NTN2TBaseMain(args)
    elif args.prediction == 'nt2n_tail':
        main = NT2NTailAttentionMain(args)
    elif args.prediction == 'ntn2t':
        main = NTN2TMain(args)
    elif args.prediction == 'ntn2t_tail':
        main = NTN2TTailAttentionMain(args)
    else:
        raise Exception('Not supported prediction type: {}'.format(args.prediction))

    return main


def train(args):
    get_main(args).train(args)


def evaluate(args):
    if args.saved_model is None:
        print('WARNING: Running eval without saved_model. Not a good idea')
    get_main(args).eval(args, print_every=1)


def grid_search(args):
    parameter_name = args.grid_name
    parameter_values = args.grid_values

    initial_title = args.title
    initial_save_dir = args.model_save_dir

    for p in parameter_values:
        suffix = '_grid_' + parameter_name + '_' + str(p)
        args.title = initial_title + suffix
        args.model_save_dir = initial_save_dir + suffix
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        setattr(args, parameter_name, p)

        main = get_main(args)
        main.train(args)


def visualize(args):
    main = get_main(args)
    model = main.model

    h2o = model.h2o.affine.weight
    h2o_line = torch.sum(h2o, dim=0).detach().numpy()

    draw_line_plot(h2o_line)
    visualize_tensor(h2o)


if __name__ == '__main__':
    print(torch.__version__)
    _args = parser.parse_args()
    assert _args.title is not None
    logger.should_log = _args.log

    if _args.grid_name is not None:
        grid_search(_args)
    elif _args.eval:
        evaluate(_args)
    else:
        train(_args)
