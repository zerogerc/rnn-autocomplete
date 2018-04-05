import argparse

import torch

from zerogercrnn.experiments.js.ast_level.data import ASTDataGenerator, DataReader, MockDataReader
from zerogercrnn.experiments.js.ast_level.main.n2n import nt_run_training
from zerogercrnn.experiments.js.ast_level.main.nt2nt import nttp_run_training
from zerogercrnn.experiments.js.ast_level.main.n2n_attention import nt_at_run_training
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.utils.time import logger

parser = argparse.ArgumentParser(description='AST level neural network')
parser.add_argument('--config_file', type=str, help='File with training process configuration')
parser.add_argument('--title', type=str, help='Title for this run')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--real_data', action='store_true', help='use real data?')
parser.add_argument('--log', action='store_true', help='log performance?')

"""
File to be able to train model from console. You could specify params of model in config.json file.
The location of this file is command line parameter 
"""


def create_data_generator(cfg, real_data):
    """Create DataReader with either real or fake data."""
    if real_data:
        reader = DataReader(
            file_training=cfg.train_file,
            file_eval=None,
            encoding=cfg.encoding,
            limit_train=cfg.data_train_limit,
            limit_eval=cfg.data_eval_limit,
            cuda=True
        )
    else:
        reader = MockDataReader()

    data_generator = ASTDataGenerator(
        data_reader=reader,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size
    )

    return data_generator


def main(title, cuda, real_data, cfg):
    # Data
    data_generator = create_data_generator(cfg, real_data)

    if cfg.prediction_type == 'nt':
        train_fun = nt_run_training
    elif cfg.prediction_type == 'nttp':
        train_fun = nttp_run_training
    elif cfg.prediction_type == 'nt_attention':
        train_fun = nt_at_run_training
    else:
        raise Exception('Unknown type of prediction. Should be one of: {}'.format('nt, nttp'))

    train_fun(
        cfg=cfg,
        title=title,
        cuda=cuda,
        data_generator=data_generator,
    )


if __name__ == '__main__':
    args = parser.parse_args()

    assert args.title is not None
    cuda = args.cuda
    logger.should_log = args.log
    real_data = args.real_data

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    config = Config()
    config.read_from_file(args.config_file)

    main(args.title, cuda, real_data, config)
