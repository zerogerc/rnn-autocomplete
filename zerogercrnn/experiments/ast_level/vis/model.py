import matplotlib.pyplot as plt
import numpy as np

from zerogercrnn.experiments.ast_level.vis.utils import visualize_attention_matrix

"""
Tools for model visualization.
"""


def visualize_attention(file):
    """Visualize attention matrix stored as numpy array"""
    tensor = np.load(file)
    visualize_attention_matrix(tensor)


if __name__ == '__main__':
    visualize_attention('eval_local/attention/per_depth_matrix.npy')
