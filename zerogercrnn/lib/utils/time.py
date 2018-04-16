from tqdm import tqdm
from itertools import islice
import time

# hack for tqdm
tqdm.monitor_interval = 0


class Logger:
    def __init__(self):
        self.ct = time.clock()
        self.should_log = True

    def reset_time(self):
        self.ct = time.clock()

    def log_time_s(self, label):
        self.__log_time__(label, 1)

    def log_time_ms(self, label):
        self.__log_time__(label, 1000)

    def __log_time__(self, label, multiplier):
        if self.should_log:
            print("{}: {}".format(label, multiplier * (time.clock() - self.ct)))
            self.ct = time.clock()


logger = Logger()


def tqdm_lim(iter, total=None, lim=None):
    if (total is None) and (lim is None):
        return tqdm(iter)

    right = 1000000000
    if total is not None:
        right = min(right, total)

    if lim is not None:
        right = min(right, lim)

    return tqdm(islice(iter, 0, right), total=right)
