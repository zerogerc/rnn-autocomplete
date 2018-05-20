import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn as nn

from zerogercrnn.lib.core import CombinedModule, LinearLayer, NormalizationLayer
from zerogercrnn.lib.log import tqdm_lim
from zerogercrnn.lib.metrics import TensorVisualizer2DMetrics


class MNISTClassifier(CombinedModule):

    def __init__(self, num_inputs, action_space, hidden_size1=256, hidden_size2=128):
        super(MNISTClassifier, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = self.module(LinearLayer(num_inputs, hidden_size1))
        self.linear2 = self.module(LinearLayer(hidden_size1, hidden_size2))
        self.linear3 = self.module(LinearLayer(hidden_size2, num_outputs))
        self.bn1 = self.module(NormalizationLayer(hidden_size1))
        self.bn2 = self.module(NormalizationLayer(hidden_size2))

    def forward(self, inputs):
        x = inputs
        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        out = self.linear3(x)
        return out


def get_data_loader(batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            'data/temp',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True
    )


if __name__ == '__main__':
    batch_size = 10

    train_loader = get_data_loader(batch_size)
    model = MNISTClassifier(num_inputs=28 * 28, action_space=10)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    metrics_before = TensorVisualizer2DMetrics(file='eval/temp/test_before')
    metrics_after = TensorVisualizer2DMetrics(file='eval/temp/test_after')


    def norm_hook(module, m_input, m_output):
        metrics_before.report(m_input[0])
        metrics_after.report(m_output)


    model.bn2.register_forward_hook(norm_hook)

    model.train()
    it = 0
    for i in range(1):
        for data, target in tqdm_lim(train_loader, lim=2000):
            optimizer.zero_grad()
            m_output = model(data.view(batch_size, -1))
            loss = loss_func(m_output, target)
            loss.backward()
            optimizer.step()

            if it % 1000 == 0:
                print(loss.item())
            it += 1

    metrics_before.get_current_value(should_print=True)
    metrics_after.get_current_value(should_print=False)
