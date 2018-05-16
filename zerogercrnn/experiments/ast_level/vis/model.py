import matplotlib.pyplot as plt
import numpy as np

from zerogercrnn.experiments.ast_level.vis.utils import visualize_attention_matrix, draw_line_plot

"""
Tools for model visualization.
"""


def visualize_attention(file):
    """Visualize attention matrix stored as numpy array"""
    tensor = np.load(file)
    visualize_attention_matrix(tensor)


def visualize_output_combination(file):
    tensor = np.load(file)
    print(np.sum(tensor[:1500]) / 1500)
    print(np.sum(tensor[-500:]) / 500)
    draw_line_plot(tensor)

if __name__ == '__main__':
    # visualize_attention('eval_local/attention/per_depth_matrix.npy')
    visualize_output_combination('eval/temp/output_sum_matrix.npy')
