import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.experiments.utils import filter_requires_grad


def get_optimizer_args(args, model):
    return optim.Adam(
        params=filter_requires_grad(model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )


def get_sparse_optimizer_args(args, model):
    return optim.SparseAdam(
        params=filter_requires_grad(model.sparse_parameters()),
        lr=args.learning_rate
    )


def get_optimizers(args, model):
    optimizers = []
    if len(list(filter_requires_grad(model.parameters()))) != 0:
        optimizers.append(get_optimizer_args(args, model))
    if len(list(filter_requires_grad(model.sparse_parameters()))) != 0:
        optimizers.append(get_sparse_optimizer_args(args, model))

    if len(optimizers) == 0:
        raise Exception('Model has no parameters!')

    return optimizers


def get_scheduler_args(args, optimizer):
    return MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(args.decay_after_epoch, args.epochs + 1)),
        gamma=args.decay_multiplier
    )
