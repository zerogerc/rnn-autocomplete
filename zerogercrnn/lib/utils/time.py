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