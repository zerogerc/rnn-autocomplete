import matplotlib.pyplot as plt
import numpy as np
from zerogercrnn.lib.data.preprocess import write_json, read_jsons, extract_jsons_info, JsonExtractor

FILE_STAT_TREE_HEIGHTS = 'data/ast/stat_tree_heights.json'


def tree_heights_stats(tree_heights):
    print('Min height of the tree: {}'.format(min(tree_heights)))
    print('Max height of the tree: {}'.format(max(tree_heights)))
    print('Average height of the trees: {}'.format(float(sum(tree_heights)) / len(tree_heights)))


def tree_heights_stats_from_file(tree_heights_file):
    tree_heights_stats(list(read_jsons(tree_heights_file))[0])


def tree_heights_distribution(tree_heights_file):
    tree_heights = list(read_jsons(tree_heights_file))[0]
    tree_heights = sorted(tree_heights)
    x = []
    y = []
    for i in range(len(tree_heights)):
        if (i == len(tree_heights) - 1) or (tree_heights[i] != tree_heights[i + 1]):
            x.append(tree_heights[i])
            y.append(float(i + 1) / len(tree_heights))

    plt.plot(x, y)
    plt.xlabel('Tree height')
    plt.ylabel('Percent of data')
    plt.show()


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
    for current_height in extract_jsons_info(extractor, 'data/programs_training.json'):
        tree_heights.append(current_height)

    if heights_file is not None:
        write_json(heights_file, tree_heights)
    tree_heights_stats(tree_heights=tree_heights)


def run_main():
    # calc_tree_heights(heights_file=FILE_STAT_TREE_HEIGHTS)
    tree_heights_stats_from_file(tree_heights_file=FILE_STAT_TREE_HEIGHTS)
    tree_heights_distribution(tree_heights_file=FILE_STAT_TREE_HEIGHTS)


if __name__ == '__main__':
    run_main()
