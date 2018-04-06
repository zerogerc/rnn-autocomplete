import os

import torch
from torch.autograd import Variable

from zerogercrnn.experiments.js.ast_level.data import DataReader
from zerogercrnn.experiments.js.ast_level.model.core import RecurrentCore
from zerogercrnn.experiments.js.ast_level.model.n2n import NTModel
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.utils.state import load_if_saved, load_cuda_on_cpu

MODELS_DIR = '/Users/zerogerc/Documents/gcp_models'
DATA_DIR = '/Users/zerogerc/Documents/datasets/js_dataset.tar/processed/'
RESULTS_DIR = '/Users/zerogerc/Documents/diploma/results_tree/'

MODEL_FILE = os.path.join(MODELS_DIR, '02Apr2018_n2n_last/model_epoch_39')
CONFIG_FILE = os.path.join(MODELS_DIR, '02Apr2018_n2n_last/config.json')

CUDA = False

ENCODING = 'ISO-8859-1'

TRAIN_FILE = os.path.join(DATA_DIR, 'programs_training_one_hot.json')
EVAL_FILE = os.path.join(DATA_DIR, 'programs_eval_one_hot.json')
EVAL_LIMIT = 100

RESULTS_FILE = os.path.join(RESULTS_DIR, 'eval_prediction.json')


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

    model = NTModel(
        non_terminal_vocab_size=cfg.non_terminals_count,
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


def main_save_prediction(config_file, cuda, model_file, file_eval, limit_eval):
    cfg = Config()
    cfg.read_from_file(config_file)

    model = load_model(cuda, cfg, model_file)
    reader = create_eval_reader(cuda, cfg, file_eval, limit_eval)

    f_write = open(RESULTS_FILE, mode='w', encoding=ENCODING)

    def write_variable_as_json(label, first, variable):
        f_write.write('"')
        f_write.write(label)
        f_write.write('"')
        f_write.write(':')

        f_write.write('[')
        f_write.write(str(first))
        f_write.write(',')
        f_write.write(','.join([str(x) for x in variable.data]))
        f_write.write(']')

    for program in reader.data_eval:
        target, top, = model_top_prediction_for_program(cuda, program, model)

        f_write.write('{')
        write_variable_as_json('nt_target', program.N[0], target)
        f_write.write(',')

        write_variable_as_json('nt_prediction', program.N[0], top)
        f_write.write('}\n')


def evaluate(cuda, cfg, model, reader):
    """Run model on eval dataset and calc accuracy for predicting terminal and non-terminal tokens."""
    t_correct, t_all = 0, 0
    nt_correct, nt_all = 0, 0

    for program in reader.data_eval:
        target, top, = model_top_prediction_for_program(cuda, program, model)

        # Non terminals
        nt_cur = target.size()[0]
        nt_all += nt_cur
        nt_correct += nt_cur - torch.nonzero(target - top).size()[0]

        # # Terminals
        # t_cur = target[1].size()[0]
        # t_all += t_cur
        # t_correct += t_cur - torch.nonzero(target[1] - top[1]).size()[0]

    print('Accuracy')
    print('Non terminals: {}'.format(float(nt_correct) / nt_all))
    # print('Terminals: {}'.format(float(t_correct) / t_all))


def model_output_for_program(cuda, program, model):
    non_terminal_input = Variable(program.N[:-1].unsqueeze(1).unsqueeze(2))
    # terminal_input = Variable(program.T[:-1].unsqueeze(1).unsqueeze(2))

    non_terminal_target = Variable(program.N[1:])
    # terminal_target = Variable(program.T[1:])

    if cuda:
        non_terminal_input = non_terminal_input.cuda()
        # terminal_input = terminal_input.cuda()
        non_terminal_target = non_terminal_target.cuda()
        # terminal_target = terminal_target.cuda()

    model.zero_grad()
    target = non_terminal_target
    output, hidden = model(non_terminal_input, None)

    return target, output


def model_top_prediction_for_program(cuda, program, model):
    target, output = model_output_for_program(cuda, program, model)

    # Non terminals
    _, nt_top_prediction = torch.max(output, dim=2)

    return target, nt_top_prediction[:, 0]


def model_top_tokens(cuda, program, model):
    pass


if __name__ == '__main__':
    cfg = Config()
    cfg.read_from_file(CONFIG_FILE)


    model = load_model(CUDA, cfg, MODEL_FILE)
    reader = create_eval_reader(CUDA, cfg, EVAL_FILE, EVAL_LIMIT)

    evaluate(CUDA, cfg, model, reader)
    # main_save_prediction(CONFIG_FILE, CUDA, MODEL_FILE, EVAL_FILE, EVAL_LIMIT)
