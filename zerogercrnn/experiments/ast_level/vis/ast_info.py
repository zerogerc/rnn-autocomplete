import json

import matplotlib.pyplot as plt
import numpy as np

from zerogercrnn.lib.preprocess import write_json, read_jsons, read_json, extract_jsons_info, JsonExtractor

FILE_TRAINING = 'data/programs_training.json'
FILE_TRAINING_PROCESSED = 'data/ast/file_train.json'
FILE_STAT_TREE_HEIGHTS = 'data/ast/stat_tree_heights.json'
FILE_STAT_PROGRAM_LENGTHS = 'data/ast/stat_program_lengths.json'


# region Utils

def draw_plot(x, y, x_label=None, y_label=None):
    plt.plot(x, y)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.show()


def get_percentile_plot(stat):
    """Get x and y for plot describing percentile of stat < x for each x.

    :param stat: array of int values
    """
    stat = sorted(stat)
    x = []
    y = []
    for i in range(len(stat)):
        if (i == len(stat) - 1) or (stat[i] != stat[i + 1]):
            x.append(stat[i])
            y.append(float(i + 1) / len(stat))

    return x, y


def plot_percentile_from_file(file, x_label, y_label):
    stat = list(read_jsons(file))[0]
    x, y = get_percentile_plot(stat)
    draw_plot(x, y, x_label=x_label, y_label=y_label)


# endregion

# region TreeHeight
def print_tree_heights_stats(tree_heights):
    print('Min height of the tree: {}'.format(min(tree_heights)))
    print('Max height of the tree: {}'.format(max(tree_heights)))
    print('Average height of the trees: {}'.format(float(sum(tree_heights)) / len(tree_heights)))


def print_tree_heights_stats_from_file(tree_heights_file):
    print_tree_heights_stats(list(read_jsons(tree_heights_file))[0])


def tree_heights_distribution(tree_heights_file):
    plot_percentile_from_file(tree_heights_file, x_label='Tree height', y_label='Percent of data')


class JsonTreeHeightExtractor(JsonExtractor):

    def __init__(self):
        self.buffer = {}

    def extract(self, raw_json):
        return self._calc_height(raw_json)

    def _calc_height(self, raw_json):
        to_calc = []
        for node in raw_json:
            if node == 0:
                break

            if 'children' in node:
                to_calc.append(node)
            else:
                self.buffer[int(node['id'])] = 1

        for node in reversed(to_calc):
            id = int(node['id'])
            self.buffer[id] = 0
            for children in node['children']:
                self.buffer[id] = max(self.buffer[id], self.buffer[int(children)] + 1)

        return self.buffer[0]


def calc_tree_heights(heights_file):
    tree_heights = []

    extractor = JsonTreeHeightExtractor()
    for current_height in extract_jsons_info(extractor, FILE_TRAINING):
        tree_heights.append(current_height)

    if heights_file is not None:
        write_json(heights_file, tree_heights)
    print_tree_heights_stats(tree_heights=tree_heights)


# endregion

# region ProgramLen


def plot_program_len_percentiles(lengths_file):
    plot_percentile_from_file(lengths_file, x_label='Program lengths', y_label='Percentile')


class JsonProgramLenExtractor(JsonExtractor):

    def extract(self, raw_json):
        return len(raw_json) - 1


def calc_programs_len(lengths_file):
    extractor = JsonProgramLenExtractor()

    program_lengths = list(extract_jsons_info(extractor, FILE_TRAINING))
    if lengths_file is not None:
        write_json(lengths_file, program_lengths)


# endregion


class JsonProgramDepthStatExtractor(JsonExtractor):

    def extract(self, raw_json):
        depths_prob = np.zeros(50)
        for node in raw_json:
            depths_prob[min(node['d'], 49)] += 1

        return depths_prob


def extract_depths_histogram():
    extractor = JsonProgramDepthStatExtractor()

    depths_prob = np.zeros(50)
    for info in extract_jsons_info(extractor, FILE_TRAINING_PROCESSED):
        depths_prob = depths_prob + info

    res = [x for x in depths_prob]
    with open('eval/ast/stat/node_depths.json', 'w') as f:
        f.write(json.dumps(res))


def draw_histogram(file):
    values = read_json(file)
    all = np.sum(values)
    values /= all

    plt.plot(values)
    # n, bins, patches = plt.hist(values, 100, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.show()


class EasyNonTerminalsExtractor(JsonExtractor):

    def __init__(self):
        super().__init__()
        self.parents_table = {}

    def extract(self, raw_json):
        for i in range(len(raw_json) - 1):
            node = raw_json[i]
            if 'children' in node:
                parent_type = node['type']
                children = node['children']
                for position in range(len(children)):
                    child_type = raw_json[int(children[position])]['type']

                    if child_type not in self.parents_table:
                        self.parents_table[child_type] = []

                    self.parents_table[child_type].append(parent_type + '_' + str(position))


def get_easy_non_terminals(file, lim=None):
    extractor = EasyNonTerminalsExtractor()
    for info in extract_jsons_info(extractor, file, lim=lim):
        print(info)


class NonTerminalsStatExtractor(JsonExtractor):
    def __init__(self):
        super().__init__()
        self.stat = {}

    def extract(self, raw_json):
        for i in range(len(raw_json) - 1):
            t = raw_json[i]['type']
            if t not in self.stat:
                self.stat[t] = 0

            self.stat[t] += 1

        return True


def visualize_nt_stat(file):
    stat = read_json(file)
    labels = []
    values = []
    sum = 0
    for k in sorted(stat.keys()):
        labels.append(k)
        values.append(stat[k])
        sum += stat[k]

    x = np.arange(len(values))
    y = np.array(values) / sum * 100

    plt.xticks(x, labels, rotation=30, horizontalalignment='right', fontsize=5)
    plt.grid(True)

    plt.plot(x, y)
    plt.show()


class UNKNTExtractor(JsonExtractor):

    def extract(self, raw_json):
        unk_count = 0
        for node in raw_json:
            if node['N'] == 321:
                unk_count += 1
        return len(raw_json), unk_count


def get_unk_nt_percentage(file_eval):  # unk percentage: 8.86e-7
    extractor = UNKNTExtractor()
    total_count = 0
    unk_count = 0
    for (c_t, c_u) in extract_jsons_info(extractor, file_eval):
        total_count += c_t
        unk_count += c_u

    print(float(unk_count) / total_count)


def get_non_terminals_statistic(file, lim=None):
    extractor = NonTerminalsStatExtractor()
    list(extract_jsons_info(extractor, file, lim=lim))

    with open('data/ast/stat_nt_occurrences.json', mode='w') as f:
        f.write(json.dumps(extractor.stat))


def run_main():
    get_unk_nt_percentage('data/pyast/file_eval.json')
    # extract_depths_histogram()
    # draw_histogram('eval/ast/stat/node_depths.json')

    # get_easy_non_terminals(file='data/programs_eval_10000.json', lim=100)
    # get_non_terminals_statistic(file='data/programs_eval_10000.json', lim=10000)
    # visualize_nt_stat(file='data/ast/stat_nt_occurrences.json')

    # calc_programs_len(FILE_STAT_PROGRAM_LENGTHS)
    # plot_program_len_percentiles(FILE_STAT_PROGRAM_LENGTHS)

    # calc_tree_heights(heights_file=FILE_STAT_TREE_HEIGHTS)
    # print_tree_heights_stats_from_file(tree_heights_file=FILE_STAT_TREE_HEIGHTS)
    # tree_heights_distribution(tree_heights_file=FILE_STAT_TREE_HEIGHTS)


if __name__ == '__main__':
    run_main()
