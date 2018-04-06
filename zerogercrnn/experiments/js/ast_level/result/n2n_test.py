import os

import torch.nn as nn
import torch

from zerogercrnn.experiments.js.ast_level.data import ASTDataGenerator
from zerogercrnn.experiments.js.ast_level.model.n2n import NTModel
from zerogercrnn.experiments.js.ast_level.result.common import create_eval_reader, create_recurrent_core, create_model
from zerogercrnn.experiments.js.ast_level.routine.n2n import run_model
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.utils.time import logger


def calc_accuracy(cuda, config, reader, model):
    generator = ASTDataGenerator(
        data_reader=reader,
        seq_len=config.seq_len,
        batch_size=config.batch_size
    )

    hidden = None

    criterion = get_criterion(cuda)

    total_loss = 0
    loss_count = 0

    hits = 0
    misses = 0

    for eval_data in generator.get_test_generator():
        n_output, n_target, n_hidden = run_model(
            cuda=cuda,
            batch_size=config.batch_size,
            model=model,
            iter_data=eval_data,
            hidden=hidden
        )
        hidden = n_hidden

        current_loss = criterion(n_output, n_target).data[0]
        total_loss += current_loss
        loss_count += 1

        n_output = n_output[-1]
        n_target = n_target[-1]

        p_val_out, out_token = torch.max(n_output, 1)
        target_token = n_target.data[0]

        current_misses = torch.nonzero(torch.max(n_output, 1)[1] - n_target).size()[0]
        hits += n_target.size()[0] - current_misses
        misses += current_misses

        print('Hits: {}, Misses: {}'.format(n_target.size()[0] - current_misses, current_misses))

    print('Loss: {}'.format(total_loss / loss_count))
    print('Accuracy: {}'.format(hits / (hits + misses)))


def main(cuda, config_file, model_file, eval_file, eval_limit, task):
    config = Config()
    config.read_from_file(config_file)

    reader = create_eval_reader(cuda, config, eval_file, eval_limit)
    model = load_model(cuda, config, model_file)

    if task == 'accuracy':
        calc_accuracy(cuda, config, reader, model)
    else:
        raise Exception('Unknown task type: {}'.format(task))


def load_model(cuda, cfg, model_path):
    # Model
    recurrent_core = create_recurrent_core(cuda, cfg, model_type='lstm')

    return create_model(cuda, cfg, model_path, constructor=lambda: NTModel(
        non_terminal_vocab_size=cfg.non_terminals_count,
        embedding_size=cfg.embedding_size,
        recurrent_layer=recurrent_core
    ))


def get_criterion(cuda):
    base_criterion = nn.NLLLoss()

    if cuda:
        base_criterion = base_criterion.cuda()

    def criterion(n_output, n_target):
        """Expect n_output and n_target to be pair of (N, T).
            Return loss as a sum of NLL losses for non-terminal(N) and terminal(T).
        """
        sz_non_terminal = n_output.size()[-1]
        # flatten tensors to compute NLLLoss
        loss_non_terminal = base_criterion(
            n_output.view(-1, sz_non_terminal),
            n_target.view(-1)
        )

        return loss_non_terminal

    return criterion


if __name__ == '__main__':
    logger.should_log = False

    _model_dir = '/Users/zerogerc/Documents/gcp_models/03Apr2018_lstm'
    _model_file = os.path.join(_model_dir, 'model_epoch_39')
    _config_file = os.path.join(_model_dir, 'config.json')
    _eval_file = '/Users/zerogerc/Documents/datasets/js_dataset.tar/processed/programs_eval_one_hot.json'
    _eval_limit = 1000

    main(False, _config_file, _model_file, _eval_file, _eval_limit, task='accuracy')
