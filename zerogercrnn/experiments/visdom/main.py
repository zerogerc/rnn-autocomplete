import visdom
import numpy as np

from threading import Timer

cid = 0


class Plotter:
    def __init__(self):
        self.id = 30
        # self.win = win

    def append(self, vis):
        vis.line(
            X=np.arange(self.id, self.id + 10),
            Y=np.random.rand(10),
            win='expPlot1',
            name='Experiment 1',
            update='append'
        )
        vis.line(
            X=np.arange(self.id, self.id + 10),
            Y=np.random.rand(10),
            win='expPlot1',
            name='Experiment 2',
            update='append'
        )
        self.id += 10


if __name__ == '__main__':
    vis = visdom.Visdom()

    # win = vis.line(
    #     X=np.arange(0, 10),
    #     Y=np.random.rand(10)
    # )
    # vis.line(
    #     X=np.arange(10, 20),
    #     Y=np.random.rand(10),
    #     win=win,
    #     name='xx',
    # )
    # vis.line(
    #     X=np.arange(10, 20),
    #     Y=np.random.rand(10),
    #     win=win,
    #     name='xxx',
    # )

    xData=np.array([[1, 1], [2, 2], [3, 3]])
    yData=np.array([[1, 1], [2, 4], [3, 9]])
    if not vis.win_exists('expPlot1'):
        vis.line(X=xData, Y=yData, win='expPlot1', opts=dict(title='xxx', legend=['Experiment 1', 'Experiment 2']))
    # else:
    #     vis.line(X=xData, Y=yData, win='expPlot', name='Experiment 1', update=True)

    # Exp 2
    # if not vis.win_exists('expPlot'):
    #     vis.line(X=xData, Y=yData, win='expPlot', opts=dict(legend=['Experiment 2']))
    # else:
    #     vis.line(X=xData, Y=yData, win='expPlot', name='Experiment 2', update=True)

    plotter = Plotter()


    def do_append():
        plotter.append(vis)
        Timer(2, do_append, ()).start()


    # Timer(1, do_append, ()).start()
