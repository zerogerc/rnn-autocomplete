import torch
import torch.nn as nn


def get_device(cuda):
    return torch.device("cuda" if cuda else "cpu")


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
    """Forgets history of current hidden state."""
    if type(h) == torch.Tensor:
        return h.detach().requires_grad_(h.requires_grad)
    else:
        return tuple(repackage_hidden(v) for v in h)


def forget_hidden_partly_lstm_cell(h, forget_vector):
    return h[0].mul(forget_vector), h[1].mul(forget_vector)


def forget_hidden_partly(h, forget_vector):
    if type(h) == torch.Tensor:
        return h.mul(forget_vector.unsqueeze(0))  # TODO: check
    else:
        return tuple(forget_hidden_partly(v, forget_vector) for v in h)


def setup_tensor(tensor, cuda):
    return tensor.to(get_device(cuda))


def filter_requires_grad(parameters):
    return filter(lambda p: p.requires_grad, parameters)


if __name__ == '__main__':
    h1 = torch.randn((1, 8, 10))
    zeros = torch.ones(8, 1)
    zeros[1][0] = 0
    zeros[2][0] = 0

    h1.mul(zeros, out=h1)

    print(h1)
