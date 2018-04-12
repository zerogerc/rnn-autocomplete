from zerogercrnn.experiments.ast_level.raw_data import TokensRetriever, JsonConverter, OneHotConverter, ENCODING

LIM = 1000

"""
Script that downloads JS dataset and forms one-hot sequences of (N, T).

Train file: ./data/file_train.json
Eval file: ./data/file_eval.json
"""


def get_tokens(file_train):
    non_terminal_file = './data/non_terminals.txt'
    terminal_file = './data/terminals.txt'

    TokensRetriever().get_and_write_tokens(
        dataset=file_train,
        non_terminal_dest=non_terminal_file,
        terminal_dest=terminal_file,
        encoding=ENCODING,
        append_eof=True,
        lim=LIM
    )

    return non_terminal_file, terminal_file


def convert_files(file_train, file_eval, terminals_file):
    converted_train = './data/train_converted.json'
    converted_eval = './data/eval_converted.json'

    print('Train')
    JsonConverter.convert_file(
        raw_file=file_train,
        dest_file=converted_train,
        terminals_file=terminals_file,
        encoding=ENCODING,
        append_eof=True,
        lim=LIM
    )

    print('Eval')
    JsonConverter.convert_file(
        raw_file=file_eval,
        dest_file=converted_eval,
        terminals_file=terminals_file,
        encoding=ENCODING,
        append_eof=True,
        lim=LIM
    )

    return converted_train, converted_eval


def form_one_hot(converted_train, converted_eval, non_terminals_file, terminals_file):
    file_train_one_hot = './data/file_train.json'
    file_eval_one_hot = './data/file_eval.json'

    converter = OneHotConverter(
        file_non_terminals=non_terminals_file,
        file_terminals=terminals_file,
        encoding=ENCODING
    )

    print('Train')
    converter.convert_file(
        src_file=converted_train,
        dst_file=file_train_one_hot,
        lim=LIM
    )

    print('Eval')
    converter.convert_file(
        src_file=converted_eval,
        dst_file=file_eval_one_hot,
        lim=LIM
    )

    return file_train_one_hot, file_eval_one_hot


def main():
    file_train_raw = './data/programs_training.json'
    file_eval_raw = './data/programs_eval.json'

    print('Retrieving tokens ...')
    non_terminals_file, terminals_file = get_tokens(
        file_train=file_train_raw
    )

    print('Converting to sequences ...')
    converted_train, converted_eval = convert_files(
        file_train=file_train_raw,
        file_eval=file_eval_raw,
        terminals_file=terminals_file
    )

    print('Forming one-hot ...')
    file_train_one_hot, file_eval_one_hot = form_one_hot(
        converted_train=converted_train,
        converted_eval=converted_eval,
        non_terminals_file=non_terminals_file,
        terminals_file=terminals_file
    )

    print('Train file: {}'.format(file_train_one_hot))
    print('Eval file: {}'.format(file_eval_one_hot))


if __name__ == '__main__':
    main()
