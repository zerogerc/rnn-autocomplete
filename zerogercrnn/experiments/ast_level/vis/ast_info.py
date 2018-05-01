import matplotlib.pyplot as plt

from zerogercrnn.lib.data.preprocess import write_json, read_jsons, extract_jsons_info, JsonExtractor

FILE_TRAINING = 'data/programs_training.json'
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

def run_main():
    # calc_programs_len(FILE_STAT_PROGRAM_LENGTHS)
    plot_program_len_percentiles(FILE_STAT_PROGRAM_LENGTHS)

    # calc_tree_heights(heights_file=FILE_STAT_TREE_HEIGHTS)
    # print_tree_heights_stats_from_file(tree_heights_file=FILE_STAT_TREE_HEIGHTS)
    # tree_heights_distribution(tree_heights_file=FILE_STAT_TREE_HEIGHTS)


if __name__ == '__main__':
    run_main()
