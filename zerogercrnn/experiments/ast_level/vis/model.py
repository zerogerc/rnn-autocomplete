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


def draw_1d_plot_from_file(file):
    line = np.load(file)
    plt.plot(line)
    plt.show()


if __name__ == '__main__':
    # visualize_line('eval/temp/layered_input_matrix.npy')
    draw_1d_plot_from_file('eval/temp/attention_matrix.npy')
    # visualize_output_combination(
    #     file_before='eval/temp/output_sum_before_matrix.npy',
    #     file_after='eval/temp/output_sum_after_matrix.npy'
    # )
    # visualize_output_combination(
    #     file_before='eval/temp/layered_input_matrix_before.npy',
    #     file_after='eval/temp/layered_input_matrix_after.npy'
    # )
    # visualize_running_mean_and_variance(
    #     mean_file='eval/temp/running_mean.npy',
    #     variance_file='eval/temp/running_var.npy'
    # )
