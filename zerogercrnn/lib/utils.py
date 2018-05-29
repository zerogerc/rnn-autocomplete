import torch
import torch.nn as nn


def get_best_device():
    """Return cuda device if cuda is available."""
    return get_device(torch.cuda.is_available())


def get_device(cuda):
    return torch.device("cuda" if cuda else "cpu")


def init_recurrent_layers(*layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


def init_layers_uniform(min_value, max_value, layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            nn.init.uniform_(param, min_value, max_value)


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


def setup_tensor(tensor):
    return tensor.to(get_best_device())


def filter_requires_grad(parameters):
    return filter(lambda p: p.requires_grad, parameters)


def register_forward_hook(module, metrics, picker):
    module.register_forward_hook(lambda _, m_input, m_output: metrics.report(picker(m_input, m_output)))


def register_output_hook(module, metrics, picker=None):
    if picker is None:
        picker = lambda m_output: m_output
    register_forward_hook(module, metrics, lambda m_input, m_output: picker(m_output))


def register_input_hook(module, metrics, picker=None):
    if picker is None:
        picker = lambda m_input: m_input[0]
    register_forward_hook(module, metrics, lambda m_input, m_output: picker(m_input))


if __name__ == '__main__':
    h1 = torch.randn((1, 8, 10))
    zeros = torch.ones(8, 1)
    zeros[1][0] = 0
    zeros[2][0] = 0

    h1.mul(zeros, out=h1)

    print(h1)
