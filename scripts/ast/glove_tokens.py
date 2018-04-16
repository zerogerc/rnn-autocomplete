from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser(description='Data processing for token level neural network')
parser.add_argument('--task', type=str, help='One of: terminals, non-terminals')
parser.add_argument('--input_file', type=str, help='Input file for task')
parser.add_argument('--output_file', type=str, help='Output file for task')
parser.add_argument('--token_map_file', type=str, help='Map from token name to int file')

LIM = 100000
ENCODING = 'ISO-8859-1'
EMP_TOKEN = '<emp>'


def write_map(file, raw_map):
    f_write = open(file, mode='w', encoding=ENCODING)
    f_write.write(json.dumps(raw_map))


def create_terminals_file(args, lim=LIM):
    """Create file for terminals consisiting of sequence of token numbers. Each token is mapped into it's number.
        i.e. data.x.y -> 0 1 2 1 3

        NB: token numbers can have a big values. (Number of different tokens in data)
        No unk tokens here.
    """

    f_write = open(args.output_file, mode='w', encoding=ENCODING)

    terminals = {EMP_TOKEN: 0}
    current_id = 1

    it = 0
    with open(args.input_file, mode='r', encoding=ENCODING) as f:
        for l in tqdm(f, total=min(lim, 100000)):
            it += 1

            raw_json = json.loads(l)
            converted = []
            for node in raw_json:
                if node == 0:
                    break

                # add terminal
                if 'value' in node:
                    node_value = str(node['value'])
                    if node_value not in terminals.keys():
                        terminals[node_value] = current_id
                        current_id += 1
                else:
                    node_value = EMP_TOKEN

                converted.append(terminals[node_value])

            f_write.write(' '.join([str(x) for x in converted]))
            f_write.write(' ')

            if (lim is not None) and (it == lim):
                break

    write_map(args.token_map_file, terminals)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.task == 'terminals':
        create_terminals_file(args)
    else:
        raise Exception('Not supported task')
