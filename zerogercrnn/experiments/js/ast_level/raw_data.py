import json
import os

from tqdm import tqdm

"""Utils for parsing and providing dataset from here: https://www.srl.inf.ethz.ch/js150.php."""

DIR_DATASET = '/Users/zerogerc/Documents/datasets/js_dataset.tar'

FILE_TRAINING_DATASET = os.path.join(DIR_DATASET, 'programs_training.json')
FILE_EVAL_DATASET = os.path.join(DIR_DATASET, 'programs_eval.json')

FILE_TRAINING_PROCESSED = os.path.join(DIR_DATASET, 'programs_processed_training.json')
FILE_EVAL_PROCESSED = os.path.join(DIR_DATASET, 'programs_processed_eval.json')

FILE_TRAINING_ONE_HOT = os.path.join(DIR_DATASET, 'programs_training_one_hot.json')
FILE_EVAL_ONE_HOT = os.path.join(DIR_DATASET, 'programs_eval_one_hot.json')

FILE_NON_TERMINAL_TOKENS = os.path.join(DIR_DATASET, 'non_terminal_tokens.txt')
FILE_TERMINAL_TOKENS = os.path.join(DIR_DATASET, 'terminal_tokens.txt')

# ENCODING = 'utf-8'
# ENCODING = 'latin-1'
ENCODING = 'ISO-8859-1'

EMPTY_TOKEN = '_EMP_'  # token means that for particular terminal there are no corresponding non-terminal
UNKNOWN_TOKEN = '_UNK_'  # token means that non-terminal token is rare
EOF_TOKEN = '_EOF_'  # token indicating end of program


class OneHotConverter:
    """Converts tokenized sequence from JsonConverter into one-hot using token files provided by TokenRetriever."""

    def __init__(self, file_terminals, file_non_terminals, encoding=ENCODING):
        # dataset was already converted using this limit.
        # TODO: do not use limit when converting dataset

        self.terminals, self.terminal_idx = DataUtils.read_terminals(
            file=file_terminals,
            limit=50000,
            encoding=encoding
        )

        self.non_terminals, self.non_terminal_idx = DataUtils.read_non_terminals(
            file=file_non_terminals,
            encoding=encoding
        )

        self.encoding = encoding

    def convert_file(self, src_file, dst_file, lim=None):
        f_read = open(src_file, mode='r', encoding=self.encoding)
        f_write = open(dst_file, mode='w', encoding=self.encoding)

        c = 0
        for l in tqdm(f_read, total=100000):
            c += 1
            raw_json = json.loads(l)

            converted_json = []
            for node in raw_json:
                if node == 0:
                    break

                N = node['N']
                T = node['T']

                if N not in self.non_terminal_idx.keys():
                    raise Exception('Unknown non terminal: {}'.format(N))
                if T not in self.terminal_idx.keys():
                    raise Exception('Unknown terminal: {}'.format(T))

                converted_json.append({
                    'N': self.non_terminal_idx[N],
                    'T': self.terminal_idx[T]
                })

            f_write.write(json.dumps(converted_json))
            f_write.write('\n')

            if (lim is not None) and (c == lim):
                break


class JsonConverter:
    """Converts raw json of parsed AST to sequence of (N, T) where N means non-terminal and T means corresponding terminal.
    T should employ coding from TokensRetriever (two bits appended).
    Output format - json like that: [{"N": "Program01", "T": "_EMP_"}, 0]"""

    @staticmethod
    def convert_file(raw_file, dest_file, terminals_file, encoding=ENCODING, append_eof=True, lim=None):
        f_read = open(raw_file, mode='r', encoding=encoding)
        f_write = open(dest_file, mode='w', encoding=encoding)
        terminals = DataUtils.read_lines(file=terminals_file, limit=50000)

        c = 0
        for l in tqdm(f_read, total=min(lim, 100000)):
            c += 1
            raw_json = json.loads(l)
            converted_json = JsonConverter._convert_json_(raw_json, terminals, append_eof)

            converted_json_string = json.dumps(converted_json)
            f_write.write(converted_json_string)
            f_write.write('\n')

            if (lim is not None) and (c == lim):
                break

    @staticmethod
    def _convert_json_(raw_json, terminals, append_eof):
        left_child, right_sibling = DataUtils.get_left_child_right_sibling(
            raw_json=raw_json,
            append_eof=append_eof
        )

        output_json = []
        for node in raw_json:
            if node == 0:
                break

            # non-terminal
            N = DataUtils.encode_non_terminal(node, left_child, right_sibling)

            # terminal
            if 'value' not in node:
                T = EMPTY_TOKEN
            elif node['value'] not in terminals:
                T = UNKNOWN_TOKEN
            else:
                T = node['value']

            output_json.append({
                'N': N,
                'T': T
            })

        if append_eof:
            output_json.append({
                'N': EOF_TOKEN,
                'T': EMPTY_TOKEN
            })

        return output_json


class TokensRetriever:
    """Process raw dataset and forms files with terminal and non-terminal tokens.
    Two bits will be appended to each non-terminal token in order to encode whether it has left child and right sibling.
    """

    def __init__(self):
        self.non_terminals = {}
        self.terminals = {}

    def get_and_write_tokens(
            self,
            dataset,
            non_terminal_dest,
            terminal_dest,
            encoding=ENCODING,
            append_eof=True,
            lim=None
    ):
        c = 0
        with open(dataset, mode='r', encoding=ENCODING) as f:
            for l in tqdm(f, total=min(lim, 100000)):
                c += 1
                self._process_single_json_(json.loads(l), append_eof=append_eof)

                if (lim is not None) and (c == lim):
                    break

        with open(non_terminal_dest, mode='w', encoding=encoding) as f:
            for t in self.non_terminals.keys():
                f.write('{}\n'.format(t))

        with open(terminal_dest, mode='w', encoding=encoding) as f:
            sorted_terminals = sorted(self.terminals.keys(), key=lambda key: self.terminals[key], reverse=True)
            for t in sorted_terminals:
                f.write('{}\n'.format(t))

    def _process_single_json_(self, raw_json, append_eof):
        left_child, right_sibling = DataUtils.get_left_child_right_sibling(
            raw_json=raw_json,
            append_eof=append_eof
        )

        for node in raw_json:
            if node == 0:
                break

            # add non-terminal
            node_type = DataUtils.encode_non_terminal(node, left_child, right_sibling)
            if node_type not in self.non_terminals.keys():
                self.non_terminals[node_type] = 0
            self.non_terminals[node_type] += 1

            # add terminal
            if 'value' in node:
                node_value = node['value']
                if node_value not in self.terminals.keys():
                    self.terminals[node_value] = 0
                self.terminals[node_value] += 1


class DataUtils:
    @staticmethod
    def read_terminals(file, limit, encoding=ENCODING):
        terminals = DataUtils.read_lines(file=file, limit=limit, encoding=encoding)
        terminals.append(EMPTY_TOKEN)
        terminals.append(UNKNOWN_TOKEN)

        terminal_idx = {}
        for i in range(len(terminals)):
            terminal_idx[terminals[i]] = i

        return terminals, terminal_idx

    @staticmethod
    def read_non_terminals(file, encoding=ENCODING):
        non_terminals = DataUtils.read_lines(file=file, limit=None, encoding=encoding)
        non_terminals.append(EOF_TOKEN)

        non_terminal_idx = {}
        for i in range(len(non_terminals)):
            non_terminal_idx[non_terminals[i]] = i

        return non_terminals, non_terminal_idx

    @staticmethod
    def read_lines(file, limit, encoding=ENCODING):
        """Read first *limit* lines from file and returns list of them."""
        lines = []
        with open(file, mode='r', encoding=encoding) as f:
            times = limit

            for l in tqdm(f, total=limit):
                if l[-1] == '\n':
                    lines.append(l[:-1])
                else:
                    lines.append(l)

                if times is not None:
                    times -= 1
                    if times == 0:
                        break

        return lines

    @staticmethod
    def get_left_child_right_sibling(raw_json, append_eof=True):
        """
        :param     raw_json: AST json employed format from here: https://www.srl.inf.ethz.ch/js150.php
        :param   append_eof: whether to append EOF token to the end of program
        :return: left_child - ids of nodes that have left child </br>
                 right_sibling - ids of nodes that have right sibling
        """
        left_child = set()
        right_sibling = set()

        for node in raw_json:
            if node == 0:
                break

            if 'children' in node:  # this node has children
                has_right_sibling_count = len(node['children']) - 1  # all except of last has right sibling
                if append_eof and node['type'] == 'Program':  # we append eof, so last children also has right sibling
                    has_right_sibling_count = len(node['children'])

                left_child.add(node['id'])  # has left child
                for i in range(has_right_sibling_count):
                    right_sibling.add(node['children'][i])

        return left_child, right_sibling

    @staticmethod
    def encode_non_terminal(node, left_child, right_sibling):
        """Encodes non-terminal based on whether it has left_child and right_sibling."""
        node_type = node['type']
        if node['id'] in left_child:
            node_type += '1'
        else:
            node_type += '0'

        if node['id'] in right_sibling:
            node_type += '1'
        else:
            node_type += '0'

        return node_type


if __name__ == '__main__':
    c = 2

    if c == 0:
        TokensRetriever().get_and_write_tokens(
            dataset=FILE_TRAINING_DATASET,
            non_terminal_dest=FILE_NON_TERMINAL_TOKENS,
            terminal_dest=FILE_TERMINAL_TOKENS,
            encoding=ENCODING,
            append_eof=True
        )
    elif c == 1:
        JsonConverter.convert_file(
            raw_file=FILE_TRAINING_DATASET,
            dest_file=FILE_TRAINING_PROCESSED,
            terminals_file=FILE_TERMINAL_TOKENS,
            encoding=ENCODING,
            append_eof=True
        )
    elif c == 2:
        converter = OneHotConverter(
            file_terminals=FILE_TERMINAL_TOKENS,
            file_non_terminals=FILE_NON_TERMINAL_TOKENS,
            encoding=ENCODING
        )
        # converter.convert_file(
        #     src_file=FILE_EVAL_PROCESSED,
        #     dst_file=FILE_EVAL_ONE_HOT
        # )
        converter.convert_file(
            src_file=FILE_TRAINING_PROCESSED,
            dst_file=FILE_TRAINING_ONE_HOT
        )
