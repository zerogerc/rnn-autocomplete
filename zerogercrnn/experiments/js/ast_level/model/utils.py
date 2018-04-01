import torch.nn as nn


def init_recurrent_layers(*layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)


def init_uniform_layers(min_value, max_value, layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            nn.init.uniform(param, min_value, max_value)
