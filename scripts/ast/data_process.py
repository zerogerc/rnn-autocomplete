import json
import argparse

from zerogercrnn.experiments.ast_level.raw_data import TokensRetriever, JsonConverter, OneHotConverter, ENCODING

parser = argparse.ArgumentParser(description='Data processing for token level neural network')
parser.add_argument('--file_train_raw', type=str, help='Raw train file')
parser.add_argument('--file_eval_raw', type=str, help='Raw eval file')
parser.add_argument('--file_non_terminals', type=str, help='File to store terminals')
parser.add_argument('--file_terminals', type=str, help='File to store non-terminals')
parser.add_argument('--file_train_converted', type=str, help='Sequence train file')
parser.add_argument('--file_eval_converted', type=str, help='Sequence eval file')
parser.add_argument('--file_train', type=str, help='One-hot train file')
parser.add_argument('--file_eval', type=str, help='One-hot eval file')
parser.add_argument('--file_glove_map', type=str, help='File from glove_tokens.py stroing map from token to number')
parser.add_argument('--file_glove_vocab', type=str, help='Vocabulary of trained Glove vectors')
parser.add_argument('--file_glove_terminals', type=str, help='Where to put terminals file of Glove')
parser.add_argument('--file_glove_vectors', type=str, help='File with Glove embedding ')
parser.add_argument('--file_glove_terminal_emb', type=str, help='Where to put one-hot terminal embeddings of Glove')

LIM = 1000000

"""
Script that forms one-hot sequences of (N, T) from JS dataset.
"""
def create_glove_terminals_file(args):
    json_data = open(args.file_glove_map).read()
    term2id = json.loads(json_data)
    id2term = {}
    for (k, v) in term2id.items():
        id2term[v] = k

    terminals = []
    with open(args.file_glove_vocab, mode='r', encoding=ENCODING) as f:
        for line in f:
            t = line.split(' ')
            terminals.append(id2term[int(t[0])])

    glove_terminals = open(args.file_glove_terminals, mode='w', encoding=ENCODING)
    glove_terminals.write('\n'.join(terminals))

def get_tokens(args):
    TokensRetriever().get_and_write_tokens(
        dataset=args.file_train_raw,
        non_terminal_dest=args.file_non_terminals,
        terminal_dest=args.file_glove_terminals,
        encoding=ENCODING,
        append_eof=True,
        lim=LIM
    )


def convert_files(args):
    print('Train')
    JsonConverter.convert_file(
        raw_file=args.file_train_raw,
        dest_file=args.file_train_converted,
        terminals_file=args.file_glove_terminals,
        encoding=ENCODING,
        append_eof=True,
        lim=LIM
    )

    print('Eval')
    JsonConverter.convert_file(
        raw_file=args.file_eval_raw,
        dest_file=args.file_eval_converted,
        terminals_file=args.file_glove_terminals,
        encoding=ENCODING,
        append_eof=True,
        lim=LIM
    )


def form_one_hot(args):
    converter = OneHotConverter(
        file_non_terminals=args.file_non_terminals,
        file_terminals=args.file_glove_terminals,
        encoding=ENCODING
    )

    print('Train')
    converter.convert_file(
        src_file=args.file_train_converted,
        dst_file=args.file_train,
        lim=LIM
    )

    print('Eval')
    converter.convert_file(
        src_file=args.file_eval_converted,
        dst_file=args.file_eval,
        lim=LIM
    )


def main():
    args = parser.parse_args()

    print('Retrieving Glove terminals')
    create_glove_terminals_file(args)

    print('Retrieving tokens ...')
    get_tokens(args)

    print('Converting to sequences ...')
    convert_files(args)

    print('Forming one-hot ...')
    form_one_hot(args)

    print('Train file: {}'.format(args.file_train))
    print('Eval file: {}'.format(args.file_eval))


if __name__ == '__main__':
    main()