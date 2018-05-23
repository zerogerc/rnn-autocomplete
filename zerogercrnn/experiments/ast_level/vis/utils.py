import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt


def visualize_attention_matrix(matrix):
    plt.matshow(matrix)
    plt.colorbar()
    plt.show()


def draw_line_plot(line):
    plt.plot(line)
    plt.show()


def visualize_tensor(tensor_to_visualize):
    """Draws a heatmap of tensor."""
    tensor_to_visualize = tensor_to_visualize.detach().numpy()
    X = np.arange(0, tensor_to_visualize.shape[0])
    Y = np.arange(0, tensor_to_visualize.shape[1])
    X, Y = np.meshgrid(X, Y, indexing='ij')

    plt.figure()
    plt.pcolor(X, Y, tensor_to_visualize)
    plt.colorbar()
    plt.show()