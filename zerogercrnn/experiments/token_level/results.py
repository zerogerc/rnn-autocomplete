import argparse
import random

import numpy
import torch

from zerogercrnn.experiments.token_level.main import create_data_generator, create_model
from zerogercrnn.experiments.token_level.main import run_model
from zerogercrnn.lib.file import read_lines, load_if_saved, load_cuda_on_cpu
from zerogercrnn.lib.visualization.text import show_diff

parser = argparse.ArgumentParser(description='AST level neural network')
parser.add_argument('--title', type=str, help='Title for this run. Used in tensorboard and in saving of models.')
parser.add_argument('--train_file', type=str, help='File with training data')
parser.add_argument('--eval_file', type=str, help='File with eval data')
parser.add_argument('--embeddings_file', type=str, help='File with embedding vectors')
parser.add_argument('--data_limit', type=int, help='How much lines of data to process (only for fast checking)')
parser.add_argument('--model_save_dir', type=str, help='Where to save trained models')
parser.add_argument('--saved_model', type=str, help='File with trained model if not fresh train')
parser.add_argument('--log', action='store_true', help='Log performance?')
parser.add_argument('--vocab', type=str, help='Vocabulary of used tokens')

parser.add_argument('--tokens_count', type=int, help='All possible tokens count')
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

ENCODING = 'ISO-8859-1'


def load_model(args, model):
    if args.saved_model is not None:
        if torch.cuda.is_available():
            load_if_saved(model, args.saved_model)
        else:
            load_cuda_on_cpu(model, args.saved_model)


def load_dictionary(tokens_path):
    id2token = read_lines(tokens_path, encoding=ENCODING)
    token2id = {}
    for id, token in enumerate(id2token):
        token2id[token] = id

    return token2id, id2token


def single_data_prediction(args, model, iter_data, hidden):
    prediction, target, hidden = run_model(model=model, iter_data=iter_data, hidden=hidden, batch_size=args.batch_size)
    return prediction, target, hidden


def get_token_for_print(id2token, id):
    if id == 0:
        return 'UNK'
    else:
        return id2token[id - 1]


def print_results_for_current_prediction(id2token, prediction, target):
    prediction_values, prediction = torch.max(prediction, dim=2)
    prediction = prediction.view(-1)
    target = target.view(-1)

    text_actual = []
    text_predicted = []

    for i in range(len(prediction)):
        is_true = prediction.data[i] == target.data[i]

        text_actual.append(get_token_for_print(id2token, target.data[i]))
        text_predicted.append(get_token_for_print(id2token, prediction.data[i]))

    return text_actual, text_predicted


def format_text(text):
    formatted = []
    it = 0
    for t in text:
        if it % 20 == 0:
            formatted.append('\n')
        it += 1
        formatted.append(t)
        formatted.append(' ')

    return formatted


def print_prediction(args):
    model = create_model(args)

    if args.batch_size != 1:
        raise Exception('batch_size should be 1 for visualization')

    if args.saved_model is not None:
        if torch.cuda.is_available():
            load_if_saved(model, args.saved_model)
        else:
            load_cuda_on_cpu(model, args.saved_model)

    generator = create_data_generator(args)

    model.eval()
    hidden = None

    lim = 1
    it = 0

    token2id, id2token = load_dictionary(args.vocab)
    text_actual = []
    text_predicted = []
    for iter_data in generator.get_eval_generator():
        prediction, target, n_hidden = single_data_prediction(args, model, iter_data, hidden)
        c_a, c_p = print_results_for_current_prediction(id2token, prediction, target)
        text_actual += c_a
        text_predicted += c_p

        hidden = n_hidden
        it += 1
        if it == lim:
            break

    show_diff(format_text(text_predicted), format_text(text_actual))


if __name__ == '__main__':
    # good seeds: 10 5 11
    # random.seed(seed)
    # numpy.random.seed(seed)

    _args = parser.parse_args()

    for seed in [13, 14, 15, 16, 17, 18, 19, 20]:
        random.seed(seed)
        numpy.random.seed(seed)
        print_prediction(_args)
