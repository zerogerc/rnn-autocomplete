import time
import math


def time_since(since):
    """
    Calculates time since given time.
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
