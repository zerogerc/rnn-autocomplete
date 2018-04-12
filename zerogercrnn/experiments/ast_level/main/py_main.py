import argparse

import torch

from zerogercrnn.experiments.ast_level.data import ASTDataGenerator, DataReader, MockDataReader
from zerogercrnn.experiments.ast_level.main.n2n import nt_run_training
from zerogercrnn.experiments.ast_level.main.n2n_attention import nt_at_run_training
from zerogercrnn.experiments.ast_level.main.nt2nt import nttp_run_training
from zerogercrnn.experiments.ast_level.main.n2n_sequential import nt_seq_sum_attention_run_training
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.utils.time import logger

parser = argparse.ArgumentParser(description='AST level neural network')
parser.add_argument('--title', type=str, help='Title for this run. Used in tensorboard and in saving of models.')
parser.add_argument('--train_file', type=str, help='File with training data')
parser.add_argument('--eval_file', type=str, help='File with eval data')
parser.add_argument('--model_save_dir', type=str, help='Where to save trained models')
parser.add_argument('--saved_model', type=str, help='File with trained model if not fresh train')
parser.add_argument('--cuda', action='store_true', help='Use cuda?')
parser.add_argument('--real_data', action='store_true', help='Use real data?')
parser.add_argument('--log', action='store_true', help='Log performance?')

parser.add_argument('--non_terminals_count', type=int, help='Non terminals count')
parser.add_argument('--terminals_count', type=int, help='Terminals count')
parser.add_argument('--data_train_limit', type=int, help='How much lines of train data to process')
parser.add_argument('--data_eval_limit', type=int, help='How much lines of eval data to process')
parser.add_argument('--prediction_type', type=str, help='Prediction type for model')
parser.add_argument('--seq_len', type=int, help='Recurrent layer time unrolling')
parser.add_argument('--batch_size', type=int, help='Size of batch')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--epochs', type=int, help='Number of epochs to run model')
parser.add_argument('--decay_after_epoch', type=int, help='Multiply lr by decay_multiplier each epoch')
parser.add_argument('--decay_multiplier', type=float, help='Multiply lr by this number after decay_after_epoch')
parser.add_argument('--embedding_size', type=int, help='Size of embedding to use')
parser.add_argument('--hidden_size', type=int, help='Hidden size of recurrent part of model')
parser.add_argument('--num_layers', type=int, help='Number of recurrent layers')
parser.add_argument('--dropout', type=float, help='Dropout to apply to recurrent layer')
parser.add_argument('--weight_decay', type=float, help='Weight decay for l2 regularization')

"""
File to be able to train model from console. You could specify params of model in config.json file.
The location of this file is command line parameter 
"""

ENCODING = 'ISO-8859-1'


def create_data_generator(args, cfg):
    """Create DataReader with either real or fake data."""
    if args.real_data:
        reader = DataReader(
            file_training=args.train_file,
            file_eval=None,
            encoding=ENCODING,
            limit_train=cfg.data_train_limit,
            limit_eval=cfg.data_eval_limit,
            cuda=args.cuda
        )
    else:
        reader = MockDataReader()

    data_generator = ASTDataGenerator(
        data_reader=reader,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size
    )

    return data_generator


def main(args, cfg):
    # Data
    data_generator = create_data_generator(args, cfg)

    if cfg.prediction_type == 'nt':
        train_fun = nt_run_training
    elif cfg.prediction_type == 'nttp':
        train_fun = nttp_run_training
    elif cfg.prediction_type == 'nt_attention':
        train_fun = nt_at_run_training
    elif cfg.prediction_type == 'nt_seq_sum_attention':
        train_fun = nt_seq_sum_attention_run_training
    else:
        raise Exception('Unknown type of prediction. Should be one of: {}'.format('nt, nttp'))

    train_fun(
        cfg=cfg,
        title=args.title,
        cuda=args.cuda,
        data_generator=data_generator,
        model_save_dir=args.model_save_dir
    )


if __name__ == '__main__':
    args = parser.parse_args()

    assert args.title is not None
    logger.should_log = args.log

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    config = Config()
    config.read_from_file(args.config_file)

    main(args, config)
