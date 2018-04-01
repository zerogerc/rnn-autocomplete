import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from zerogercrnn.lib.utils.state import load_if_saved


def get_optimizers_and_schedulers(cfg, model):
    dense_optimizer = optim.Adam(params=model.dense_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sparse_optimizer = optim.SparseAdam(params=model.sparse_params, lr=cfg.learning_rate)

    dense_scheduler = MultiStepLR(
        optimizer=dense_optimizer,
        milestones=list(range(cfg.decay_after_epoch, cfg.epochs + 1)),
        gamma=cfg.decay_multiplier
    )
    sparse_scheduler = MultiStepLR(
        optimizer=sparse_optimizer,
        milestones=list(range(cfg.decay_after_epoch, cfg.epochs + 1)),
        gamma=cfg.decay_multiplier
    )

    return [dense_optimizer, sparse_optimizer], [dense_scheduler, sparse_scheduler]


def load_if_saved_from_config(cfg, model):
    if hasattr(cfg, 'saved_model'):
        load_if_saved(model, cfg.saved_model)
