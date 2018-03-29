import time


def perf_ms(t, tag=None):
    perf(t, 'ms', 1000, tag)


def perf_s(t, tag=None):
    perf(t, 's', 1, tag)


def perf(t, units, magnitude, tag=None):
    time_passed = magnitude * (time.clock() - t)
    if tag is None:
        print('PERF: {}{}'.format(time_passed, units))
    else:
        print('PERF {}: {}{}'.format(tag, time_passed, units))


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
