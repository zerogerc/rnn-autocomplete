import torch
import torch.nn as nn
from torch.autograd import Variable

from zerogercrnn.lib.utils.time import logger


def init_recurrent_layers(*layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)


def init_layers_uniform(min_value, max_value, layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            nn.init.uniform(param, min_value, max_value)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def forget_hidden_partly_lstm_cell(h, forget_vector):
    h[0].data.mul(forget_vector, out=h[0].data)
    h[1].data.mul(forget_vector, out=h[1].data)
    return h

def forget_hidden_partly(h, forget_vector):
    if type(h) == Variable:
        logger.reset_time()
        for l in range(h.size()[0]):  # layer numbers are usually 1 to 3 so for-loop is fine here.
            h.data[l].mul(forget_vector, out=h.data[l])
        logger.log_time_ms('FORGET_HIDDEN_TIME')
        return h
    else:
        return tuple(forget_hidden_partly(v, forget_vector) for v in h)


if __name__ == '__main__':
    h1 = torch.randn((1, 8, 10))
    zeros = torch.ones(8, 1)
    zeros[1][0] = 0
    zeros[2][0] = 0

    h1.mul(zeros, out=h1)

    print(h1)
