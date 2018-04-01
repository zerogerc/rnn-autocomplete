import os

import torch
from torch.autograd import Variable

from zerogercrnn.experiments.js.ast_level.data import DataReader
from zerogercrnn.experiments.js.ast_level.network_base import RecurrentCore, JSBaseModel
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.utils.state import load_if_saved, load_cuda_on_cpu

MODELS_DIR = '/Users/zerogerc/Documents/gcp_models'
DATA_DIR = '/Users/zerogerc/Documents/datasets/js_dataset.tar/processed/'

MODEL_FILE = os.path.join(MODELS_DIR, '31Mar2018/model_epoch_3')
CONFIG_FILE = os.path.join(MODELS_DIR, '31Mar2018/config.json')

CUDA = False

TRAIN_FILE = os.path.join(DATA_DIR, 'programs_training_one_hot.json')
EVAL_FILE = os.path.join(DATA_DIR, 'programs_eval_one_hot.json')
EVAL_LIMIT = 100


def load_model(cuda, cfg, model_path):
    # Model
    recurrent_core = RecurrentCore(
        input_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        model_type='gru'
    )
    if cuda:
        recurrent_core = recurrent_core.cuda()

    recurrent_core.init_hidden(1, cuda)

    model = JSBaseModel(
        non_terminal_vocab_size=cfg.non_terminals_count,
        terminal_vocab_size=cfg.terminals_count,
        embedding_size=cfg.embedding_size,
        recurrent_layer=recurrent_core
    )
    if cuda:
        model = model.cuda()

    # Model is always trained with cuda so if we evaluate model on non cuda device
    # we should call another function for loading
    if cuda:
        load_if_saved(model, model_path)
    else:
        load_cuda_on_cpu(model, model_path)

    return model


def create_eval_reader(cuda, cfg, file_eval, eval_limit):
    reader = DataReader(
        file_training=None,
        file_eval=file_eval,
        encoding=cfg.encoding,
        limit_train=None,
        limit_eval=eval_limit,
        cuda=cuda
    )

    return reader


def main(config_file, cuda, model_file, file_eval, limit_eval):
    cfg = Config()
    cfg.read_from_file(config_file)

    model = load_model(cuda, cfg, model_file)
    reader = create_eval_reader(cuda, cfg, file_eval, limit_eval)

    evaluate(cuda, cfg, model, reader)


def evaluate(cuda, cfg, model, reader):
    """Run model on eval dataset and calc accuracy for predicting terminal and non-terminal tokens."""
    t_correct, t_all = 0, 0
    nt_correct, nt_all = 0, 0

    for program in reader.data_eval:
        target, top, = model_top_prediction_for_program(cuda, program, model)

        # Non terminals
        nt_cur = target[0].size()[0]
        nt_all += nt_cur
        nt_correct += nt_cur - torch.nonzero(target[0] - top[0]).size()[0]

        # Terminals
        t_cur = target[1].size()[0]
        t_all += t_cur
        t_correct += t_cur - torch.nonzero(target[1] - top[1]).size()[0]

    print('Accuracy')
    print('Non terminals: {}'.format(float(nt_correct) / nt_all))
    print('Terminals: {}'.format(float(t_correct) / t_all))


def model_output_for_program(cuda, program, model):
    non_terminal_input = Variable(program.N[:-1].unsqueeze(1).unsqueeze(2))
    terminal_input = Variable(program.T[:-1].unsqueeze(1).unsqueeze(2))

    non_terminal_target = Variable(program.N[1:])
    terminal_target = Variable(program.T[1:])

    if cuda:
        non_terminal_input = non_terminal_input.cuda()
        terminal_input = terminal_input.cuda()
        non_terminal_target = non_terminal_target.cuda()
        terminal_target = terminal_target.cuda()

    model.zero_grad()
    target = (non_terminal_target, terminal_target)
    output = model(non_terminal_input, terminal_input)

    return target, output


def model_top_prediction_for_program(cuda, program, model):
    target, output = model_output_for_program(cuda, program, model)

    # Non terminals
    _, nt_top_prediction = torch.max(output[0], dim=2)

    # Terminals
    _, t_top_prediction = torch.max(output[1], dim=2)

    return target, (nt_top_prediction[:, 0], t_top_prediction[:, 0])


def model_top_tokens(cuda, program, model):
    pass


if __name__ == '__main__':
    main(CONFIG_FILE, CUDA, MODEL_FILE, TRAIN_FILE, EVAL_LIMIT)
