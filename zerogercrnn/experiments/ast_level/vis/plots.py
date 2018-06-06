import matplotlib.pyplot as plt
import numpy as np

from zerogercrnn.lib.constants import EOF_TOKEN
from zerogercrnn.lib.preprocess import read_json, read_jsons
from matplotlib.lines import Line2D

COLOR_BASE = '#607D8B'
COLOR_RED = '#f44336'
COLOR_GREEN = '#4CAF50'


class Plot:
    def __init__(self, data, label=None):
        self.data = data
        self.label = label


def add_nt_x_ticks(nt):
    x = np.arange(len(nt))
    plt.xticks(x, nt, rotation=30, horizontalalignment='right', fontsize=8)


def draw_per_nt_plot_inner(nt, *plots, y_label=None):
    add_nt_x_ticks(nt)
    plt.ylabel(y_label)
    plt.grid(True)

    legend = []
    for p in plots:
        cur_legend, = plt.plot(p.data, label=p.label)
        legend.append(cur_legend)

    plt.grid(True)
    plt.legend(handles=legend)
    plt.show()


def draw_per_nt_plot(file, y_label='Per NT accuracy'):
    nt, data = list(read_jsons(file))
    draw_per_nt_plot_inner(nt, Plot(data=data), y_label=y_label)


def draw_per_nt_bar_chart(nt, *plots, y_label='Per NT accuracy'):
    ind = np.arange(len(nt))
    legend_rects = []
    legend_labels = []
    width = 0.4 / len(plots)
    current_shift = -0.2
    for p in plots:
        cur = plt.bar(ind + current_shift + width / 2, p.data, width=width)
        legend_rects.append(cur[0])
        legend_labels.append(p.label)
        current_shift += width

    plt.legend(legend_rects, legend_labels)
    add_nt_x_ticks(nt)
    plt.show()


def bar_chart():
    import numpy as np
    import matplotlib.pyplot as plt

    N = 5
    menMeans = (20, 35, 30, 35, 27)
    womenMeans = (25, 32, 34, 20, 25)
    menStd = (2, 3, 4, 1, 2)
    womenStd = (3, 5, 2, 3, 3)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, menMeans, width, yerr=menStd)
    p2 = plt.bar(ind, womenMeans, width,
                 bottom=menMeans, yerr=womenStd)

    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    plt.show()


def compare_per_nt(file1, file2, y_label='New'):
    nt1, data1 = list(read_jsons(file1))
    nt2, data2 = list(read_jsons(file2))
    assert nt1 == nt2
    nt = nt1
    data1 = np.array(data1)
    data2 = np.array(data2)
    diff = data2 - data1

    ind = np.arange(len(nt))
    p1 = plt.bar(ind, data1)
    for bar in p1:
        bar.set_facecolor(COLOR_BASE)

    p2 = plt.bar(ind, diff, bottom=data1)

    for id, bar in enumerate(p2):
        if diff[id] >= 0:
            bar.set_facecolor(COLOR_GREEN)
        else:
            bar.set_facecolor(COLOR_RED)

    custom_lines = [Line2D([0], [0], color=COLOR_BASE, lw=4),
                    Line2D([0], [0], color=COLOR_GREEN, lw=4),
                    Line2D([0], [0], color=COLOR_RED, lw=4)]
    plt.legend(custom_lines, ['База', 'Улучшение', 'Ухудшение'])

    # plt.legend((p1[0], p2[0]), (file1, file2))
    add_nt_x_ticks(nt)

    plt.ylabel(y_label)
    plt.show()


def compare_per_nt_diff_only(file1, file2, y_label='New'):
    nt1, data1 = list(read_jsons(file1))
    nt2, data2 = list(read_jsons(file2))
    assert nt1 == nt2
    nt = nt1
    data1 = np.array(data1)
    data2 = np.array(data2)
    diff = data2 - data1

    ind = np.arange(len(nt))
    p1 = plt.bar(ind, (diff) * 100, width=1)

    for id, bar in enumerate(p1):
        if diff[id] >= 0:
            bar.set_facecolor(COLOR_GREEN)
        else:
            bar.set_facecolor(COLOR_RED)

    custom_lines = [
        Line2D([0], [0], color=COLOR_GREEN, lw=4),
        Line2D([0], [0], color=COLOR_RED, lw=4)
    ]
    plt.legend(custom_lines,  ['Улучшение', 'Ухудшение'])
    plt.grid(True)

    add_nt_x_ticks(nt)

    plt.ylabel(y_label)
    plt.show()


def main():
    res_file_base = 'eval_verified/nt2n_base_30k/top5_new/nt_acc_grouped.txt'
    res_file_base_attention = 'eval_verified/nt2n_base_attention_30k/nt_acc_grouped.txt'
    res_file_layered = 'eval_verified/nt2n_base_attention_plus_layered_30k/top5_new/nt_acc_grouped.txt'

    res_file_base_old = 'eval_verified/nt2n_base/nt_acc.txt'
    res_file_layered_attention_old = 'eval_verified/nt2n_layered_attention/nt_acc.txt'
    # draw_per_nt_plot(res_file_layered_attention_old)
    compare_per_nt_diff_only(res_file_base, res_file_layered, y_label='Абсолютная разница в точности (%)')


if __name__ == '__main__':
    main()
    # bar_chart()
