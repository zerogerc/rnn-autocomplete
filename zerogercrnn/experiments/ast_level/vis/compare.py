import matplotlib.pyplot as plt
import numpy as np

from zerogercrnn.lib.constants import EOF_TOKEN
from zerogercrnn.lib.preprocess import read_json, read_jsons


def compare_per_nt(file1, file2, y_label):
    nt1, res1 = list(read_jsons(file1))
    nt2, res2 = list(read_jsons(file2))
    assert nt1 == nt2

    x = np.arange(len(nt1))
    y1 = np.array(res1)
    y2 = np.array(res2)

    my_xticks = nt1
    plt.xticks(x, my_xticks, rotation=30, horizontalalignment='right', fontsize=8)
    plt.ylabel(y_label)
    plt.grid(True)

    plt.plot(x, (y2 - y1) * 100)
    plt.show()

    # print('Diff as second - first:')
    # for i in range(len(nt1)):
    #     print('{} : {}'.format(nt1[i], res2[i] - res1[i]))


def compare_per_two_plots(file1, file2, y_label):
    nt1, res1 = list(read_jsons(file1))
    nt2, res2 = list(read_jsons(file2))
    assert nt1 == nt2

    x = np.arange(len(nt1))
    y1 = np.array(res1)
    y2 = np.array(res2)

    my_xticks = nt1
    plt.xticks(x, my_xticks, rotation=30, horizontalalignment='right', fontsize=8)
    plt.ylabel(y_label)
    plt.grid(True)

    plt.plot(x, y1, 'r', x, y2, 'g')
    plt.show()

    # print('Diff as second - first:')
    # for i in range(len(nt1)):
    #     print('{} : {}'.format(nt1[i], res2[i] - res1[i]))


def run_main():
    values_base = np.array(read_json('eval/nt2n_base/nt_acc.json'))
    values_layered = np.array(read_json('eval/nt2n_layered_attention/nt_acc.json'))
    non_terminals = read_json('data/ast/non_terminals.json')
    non_terminals.append(EOF_TOKEN)

    diff = values_layered - values_base
    for i in range(len(non_terminals)):
        print('{}: {}'.format(non_terminals[i], diff[i]))


if __name__ == '__main__':
    # run_main()
    compare_per_nt(
        file1='eval_local/nt2n_base/nt_acc.txt',
        file2='eval_local/nt2n_layered_attention/nt_acc.txt',
        y_label='Gain of layered attention comparing to base model'
    )
    compare_per_two_plots(
        file1='eval_local/nt2n_base/nt_acc.txt',
        file2='eval_local/nt2n_layered_attention/nt_acc.txt',
        y_label='Accuracies: Red - base model, Green - layered model'
    )
