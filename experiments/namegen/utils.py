import torch.nn as nn
import torch.optim as optim
import time

from lib.data.batcher import Batcher
from experiments.namegen.reader import max_name_length
from lib.utils.train import time_since


def create_train_runner(network, optimizer, criterion, batcher, batch_size):
    return lambda n_iters, print_every, plot_every: \
        run_train_new(network, optimizer, criterion, batcher, batch_size, n_iters, print_every, plot_every)


def run_train_new(
        network: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.NLLLoss,
        batcher: Batcher,
        batch_size: int,
        n_iters: int,
        print_every: int,
        plot_every: int
):
    all_losses = []
    total_loss = 0  # Reset every plot_every iters

    start = time.time()

    data_map = batcher.data_map
    train_data = data_map['train'].get_batched(batch_size)
    validation_data = data_map['validation'].get_batched(batch_size)
    test_data = data_map['test'].get_batched(batch_size)

    for iter in range(1, n_iters + 1):
        network.zero_grad()
        if iter % print_every:
            input_tensor, target_tensor = next(test_data)
        elif iter % plot_every:
            input_tensor, target_tensor = next(validation_data)
        else:
            input_tensor, target_tensor = next(train_data)

        output = network(input_tensor)

        loss = 0
        for i in range(max_name_length):
            loss += criterion(output[i], target_tensor[i])

        loss.backward()
        optimizer.step()

        loss = loss.data[0] / input_tensor.size()[0]
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (time_since(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    return all_losses
