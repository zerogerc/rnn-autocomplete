import numpy as np

from zerogercrnn.lib.preprocess import read_json
from zerogercrnn.lib.constants import EOF_TOKEN

def run_main():
    values_base = np.array(read_json('eval/nt2n_base/nt_acc.json'))
    values_layered = np.array(read_json('eval/nt2n_layered_attention/nt_acc.json'))
    non_terminals = read_json('data/ast/non_terminals.json')
    non_terminals.append(EOF_TOKEN)

    diff = values_layered - values_base
    for i in range(len(non_terminals)):
        print('{}: {}'.format(non_terminals[i], diff[i]))


if __name__ == '__main__':
    run_main()