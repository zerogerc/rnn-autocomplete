import torch
import time

from tensorboardX import SummaryWriter

#  tensorboard --logdir=/tensorboard/runs
if __name__ == '__main__':
    writer = SummaryWriter('/tensorboard/runs/test')

    for step in range(10):
        dummy_s1 = torch.rand(1)
        writer.add_scalar('data/random', dummy_s1, step)
        time.sleep(1)

    # writer.add_scalar(
    #     tag='data/scalar2',
    #     scalar_value=10.,
    #     global_step=3
    # )
