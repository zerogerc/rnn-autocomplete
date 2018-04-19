import argparse
import json
import re

from tqdm import tqdm

from zerogercrnn.lib.utils.file import read_lines

parser = argparse.ArgumentParser(description='Data processing for token level neural network')
parser.add_argument('--task', type=str, help='One of: token, one_hot_json, one_hot_text')
parser.add_argument('--tokens_file', type=str, help='File with tokens')
parser.add_argument('--input_file', type=str, help='Input file for task')
parser.add_argument('--output_file', type=str, help='Output file for task')

TRAIN_FILE = '/Users/zerogerc/Documents/datasets/js_dataset.tar/programs_training_tokenized.json'
TRAIN_FILE_ONE_HOT = '/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/data/tokens/file_train.json'
TRAIN_FILE_PLAIN = '/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/data/tokens/file_train_plain.txt'

EVAL_FILE = 'data/programs_eval_tokenized.json'
EVAL_FILE_ONE_HOT = 'data/tokens/file_eval.json'
EVAL_FILE_PLAIN = 'data/tokens/file_eval_plain.txt'

TOKENS_FILE = '/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/data/tokens/tokens.txt'

ENCODING = 'ISO-8859-1'

EMPTY_STRING_SPACE = re.compile('[ \t]+')
EMPTY_STRING_NEWLINE = re.compile('[ \n\t]+')


# UNK TOKEN is zero

def normalize_token(token):
    if EMPTY_STRING_SPACE.fullmatch(token):
        return ' '
    if EMPTY_STRING_NEWLINE.fullmatch(token):
        return '\n'
    return token.strip()


def read_tokens(tokens_path):
    id2token = read_lines(tokens_path, encoding=ENCODING)
    token2id = {}
    for id, token in enumerate(id2token):
        token2id[token] = id

    return token2id, id2token


def get_tokens(file_path, output_path, lim=100):
    tokens = {}
    for l in open(file=file_path, mode='r', encoding=ENCODING):
        for t in json.loads(l):
            t = normalize_token(t)
            if t not in tokens.keys():
                tokens[t] = 0
            tokens[t] += 1

    with open(output_path, mode='w', encoding=ENCODING) as f:
        sorted_terminals = sorted(tokens.keys(), key=lambda key: tokens[key], reverse=True)
        for t in sorted_terminals[:lim]:
            f.write('{}\n'.format(t))


def convert_to_one_hot(file_path, tokens_path, output_file, total):
    token2id, id2token = read_tokens(tokens_path)

    all_tokens = 0
    unk_tokens = 0

    out_file = open(file=output_file, mode='w', encoding=ENCODING)
    for l in tqdm(open(file=file_path, mode='r', encoding=ENCODING), total=total):
        one_hot = []
        for t in json.loads(l):
            t = normalize_token(t)
            if t == ' ':  # skip spaces
                continue

            all_tokens += 1
            if t in token2id.keys():
                one_hot.append(1 + token2id[t])
            else:
                one_hot.append(0)
                unk_tokens += 1

        out_file.write(json.dumps(one_hot))
        out_file.write('\n')

    print('<unk> tokens percentage: {}'.format(float(unk_tokens) / all_tokens))


def convert_to_plain_text(file_path, tokens_path, output_file, total):
    token2id, id2token = read_tokens(tokens_path)

    out_file = open(file=output_file, mode='w', encoding=ENCODING)
    for l in tqdm(open(file=file_path, mode='r', encoding=ENCODING), total=total):
        one_hot = []
        for t in json.loads(l):
            one_hot.append(str(t))
        out_file.write(' '.join(one_hot))
        out_file.write(' ')


if __name__ == '__main__':
    args = parser.parse_args()

    # input_file = args.input_file
    # output_file = args.output_file
    # tokens_file = args.tokens_file

    input_file = EVAL_FILE
    output_file = EVAL_FILE_ONE_HOT
    tokens_file = TOKENS_FILE

    if args.task == 'token':
        get_tokens(input_file, output_file, lim=50000)
    elif args.task == 'one_hot_json':
        convert_to_one_hot(input_file, tokens_file, output_file, total=100000)
    elif args.task == 'one_hot_plain':
        convert_to_plain_text(input_file, tokens_file, output_file, total=100000)
    else:
        raise Exception('Unknown task type')
    # elif args.task == 'one_hot_text':
    #     convert_to_plain_text(TRAIN_FILE, TOKENS_FILE, )
