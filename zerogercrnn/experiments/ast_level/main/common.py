import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.lib.utils.state import load_if_saved


def get_optimizer_args(args, model):
    return optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )


def get_scheduler_args(args, optimizer):
    return MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(args.decay_after_epoch, args.epochs + 1)),
        gamma=args.decay_multiplier
    )


def get_optimizers_and_schedulers(cfg, model):
    optimizers = []
    schedulers = []

    if len(model.dense_params) != 0:
        optimizers.append(optim.Adam(params=model.dense_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay))
        schedulers.append(MultiStepLR(
            optimizer=optimizers[-1],
            milestones=list(range(cfg.decay_after_epoch, cfg.epochs + 1)),
            gamma=cfg.decay_multiplier
        ))

    if len(model.sparse_params) != 0:
        optimizers.append(optim.SparseAdam(params=model.sparse_params, lr=cfg.learning_rate))
        schedulers.append(MultiStepLR(
            optimizer=optimizers[-1],
            milestones=list(range(cfg.decay_after_epoch, cfg.epochs + 1)),
            gamma=cfg.decay_multiplier
        ))

    return optimizers, schedulers


def load_if_saved_from_config(cfg, model):
    if hasattr(cfg, 'saved_model'):
        load_if_saved(model, cfg.saved_model)
