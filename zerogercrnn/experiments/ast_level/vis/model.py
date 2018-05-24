import os

import numpy as np
from matplotlib import pyplot as plt

from zerogercrnn.experiments.ast_level.vis.utils import visualize_attention_matrix

"""
Tools for model visualization.
"""


def visualize_attention(file):
    """Visualize attention matrix stored as numpy array"""
    tensor = np.load(file)
    visualize_attention_matrix(tensor)


def visualize_output_combination(file_before, file_after):
    tensor_before = np.load(file_before)
    print(np.sum(tensor_before[:1500]) / 1500)
    print(np.sum(tensor_before[-500:]) / 500)
    plt.plot(tensor_before, 'r')

    tensor_after = np.load(file_after)
    print(np.sum(tensor_after[:1500]) / 1500)
    print(np.sum(tensor_after[-500:]) / 500)
    plt.plot(tensor_after, 'g')

    plt.show()


def visualize_line(file):
    line = np.load(file)
    plt.plot(line)
    plt.show()


def visualize_running_mean_and_variance(mean_file, variance_file):
    mean = np.load(mean_file)
    variance = np.load(variance_file)

    plt.plot(variance)
    plt.show()


def draw_1d_plot_from_file(*files):
    legend = []
    for f in files:
        cur, = plt.plot(np.load(f), label=f)
        legend.append(cur)

    plt.legend(handles=legend)
    plt.show()


def draw_mean_deviation_variance(directory='eval/temp'):
    draw_1d_plot_from_file(
        os.path.join(directory, 'mean.npy'),
        os.path.join(directory, 'deviation.npy'),
        os.path.join(directory, 'variance.npy')
    )


def draw_mean_variance(directory='eval/temp'):
    c1, = plt.plot(np.load(os.path.join(directory, 'mean.npy')), label=os.path.join(directory, 'mean.npy'))
    c2, = plt.plot(np.sqrt(np.load(os.path.join(directory, 'variance.npy'))), label=os.path.join(directory, 'std.npy'))
    plt.legend(handles=[c1,c2])
    plt.show()


if __name__ == '__main__':
    draw_mean_variance(directory='eval/temp/nt2n_base_attention_norm_before')
    draw_mean_variance(directory='eval/temp/nt2n_base_attention_norm_after')
    # visualize_attention(file='eval/temp/attention/per_depth_matrix.npy')
    # draw_mean_variance(directory='eval/temp/before_input')
    # draw_mean_variance(directory='eval/temp/after_input')
    #
    # draw_mean_variance(directory='eval/temp/before_output')
    # draw_mean_variance(directory='eval/temp/after_output')
    # draw_1d_plot_from_file('eval/temp/deviation.npy')
    # draw_1d_plot_from_file('eval/temp/variance.npy')

    # visualize_line('eval/temp/layered_input_matrix.npy')
    # visualize_attention('eval_local/attention/per_depth_matrix.npy')
    # visualize_output_combination(
    #     file_before='eval/temp/new_output_sum_before_matrix.npy',
    #     file_after='eval/temp/new_output_sum_after_matrix.npy'
    # )
    # visualize_output_combination(
    #     file_before='eval/temp/test_before.npy',
    #     file_after='eval/temp/test_after.npy'
    # )
    # visualize_running_mean_and_variance(
    #     mean_file='eval/temp/running_mean.npy',
    #     variance_file='eval/temp/running_var.npy'
    # )
