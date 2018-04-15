import argparse

import torch

parser = argparse.ArgumentParser(description='Metrics for token level model')
parser.add_argument('--metrics', type=str, help='Type of metrics to calculate')
parser.add_argument('--eval_file', type=str, help='File with eval data')
parser.add_argument('--saved_model', type=str, help='File with trained model')
parser.add_argument('--cuda', action='store_true', help='Use cuda?')

parser.add_argument('--tokens_count', type=int, help='All possible tokens count')  # 51k now
parser.add_argument('--seq_len', type=int, help='Recurrent layer time unrolling')
parser.add_argument('--batch_size', type=int, help='Size of batch')
parser.add_argument('--embedding_size', type=int, help='Size of embedding to use')
parser.add_argument('--hidden_size', type=int, help='Hidden size of recurrent part of model')
parser.add_argument('--num_layers', type=int, help='Number of recurrent layers')
parser.add_argument('--dropout', type=float, help='Dropout to apply to recurrent layer')
parser.add_argument('--weight_decay', type=float, help='Weight decay for l2 regularization')


from zerogercrnn.lib.results import AccuracyMeasurer
from zerogercrnn.experiments.token_level.data import TokensDataReader, TokensDataGenerator
from zerogercrnn.experiments.token_level.main import run_model, create_model, create_data_generator

def calc_accuracy(args):
    generator = create_data_generator(args)
    model = create_model(args)

    hidden_state = model.init_hidden(
        batch_size=args.batch_size,
        cuda=args.cuda,
        no_grad=True
    )

    for iter_data in generator.get_eval_generator():
        prediction, n_target, hidden = run_model(
            model=model,
            iter_data=iter_data,
            hidden=hidden_state,
            batch_size=args.batch_size,
            cuda=args.cuda,
            no_grad=True
        )
        hidden_state = hidden

if __name__ == '__main__':
    args = parser.parse_args()

    if args.metrics == 'accuracy':
        calc_accuracy(args)
    else:
        raise Exception('Unsupported metrics')