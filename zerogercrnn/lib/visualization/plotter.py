import os

import matplotlib.pyplot as plt
import numpy as np
import visdom
from tensorboardX import SummaryWriter

TENSORBOARD_DIR = 'tensorboard/runs/'


class Plotter:
    def on_new_point(self, label, x, y):
        pass

    def on_finish(self):
        pass


class MatplotlibPlotter(Plotter):
    def __init__(self, title):
        super(MatplotlibPlotter, self).__init__()
        self.title = title
        self.plots = {}

    def on_new_point(self, label, x, y):
        if label not in self.plots:
            self.plots[label] = PlotData()

        self.plots[label].x.append(x)
        self.plots[label].y.append(y)

    def on_finish(self):
        for label in self.plots:
            plt.plot(self.plots[label].x, self.plots[label].y, label=label)

        plt.title(self.title)
        plt.legend()
        plt.show()


class VisdomPlotter(Plotter):
    def __init__(self, title, plots):
        super(VisdomPlotter, self).__init__()
        self.title = title
        self.vis = visdom.Visdom()
        self.plots = set(plots)

        self.vis.line(
            X=np.zeros((1, len(plots))),
            Y=np.zeros((1, len(plots))),
            win=self.title,
            opts=dict(legend=plots)
        )

    def on_new_point(self, label, x, y):
        if label not in self.plots:
            raise Exception('Plot should be in plots set!')

        self.vis.line(
            X=np.array([x]),
            Y=np.array([y]),
            win=self.title,
            name=label,
            update='append'
        )


class TensorboardPlotter(Plotter):
    def __init__(self, title):
        path = os.path.join(os.getcwd(), TENSORBOARD_DIR + title)
        self.writer = SummaryWriter(path)

    def on_new_point(self, label, x, y):
        self.writer.add_scalar(
            tag=label,
            scalar_value=y,
            global_step=x
        )


class PlotData:
    def __init__(self):
        self.x = []
        self.y = []

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)


if __name__ == '__main__':
    plotter = VisdomPlotter(title='x', plots=['y', 'z'])
