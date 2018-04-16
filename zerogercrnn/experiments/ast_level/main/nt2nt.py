import torch.nn as nn

from zerogercrnn.experiments.ast_level.main.common import get_optimizers_and_schedulers, load_if_saved_from_config
from zerogercrnn.experiments.ast_level.model.nt2nt import JSBaseModel
from zerogercrnn.lib.core import RecurrentCore
from zerogercrnn.experiments.ast_level.routine.nt2nt import ASTRoutine
from zerogercrnn.lib.train.run import TrainEpochRunner


def nttp_run_training(cfg, title, cuda, data_generator, model_save_dir):
    criterion = get_criterion(cuda)

    model = nttp_get_model(cfg, cuda)
    load_if_saved_from_config(cfg, model)

    optimizers, schedulers = get_optimizers_and_schedulers(cfg, model)

    train_routine = ASTRoutine(
        network=model,
        criterion=criterion,
        optimizers=optimizers,
        cuda=cuda
    )

    validation_routine = ASTRoutine(
        network=model,
        criterion=criterion,
        cuda=cuda
    )

    runner = TrainEpochRunner(
        network=model,
        train_routine=train_routine,
        validation_routine=validation_routine,
        data_generator=data_generator,
        schedulers=schedulers,
        plotter='tensorboard',
        title=title,
        save_dir=model_save_dir
    )

    runner.run(number_of_epochs=cfg.epochs)


def nttp_get_model(cfg, cuda):
    # Model
    recurrent_core = RecurrentCore(
        input_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        model_type='gru'
    )
    if cuda:
        recurrent_core = recurrent_core.cuda()

    recurrent_core.init_hidden(cfg.batch_size, cuda)

    model = JSBaseModel(
        non_terminal_vocab_size=cfg.non_terminals_count,
        terminal_vocab_size=cfg.terminals_count,
        embedding_size=cfg.embedding_size,
        recurrent_layer=recurrent_core
    )

    if cuda:
        model = model.cuda()

    return model


def get_criterion(cuda):
    base_criterion = nn.NLLLoss()
    if cuda:
        base_criterion = base_criterion.cuda()

    def criterion(n_output, n_target):
        """Expect n_output and n_target to be pair of (N, T).
            Return loss as a sum of NLL losses for non-terminal(N) and terminal(T).
        """
        sz_non_terminal = n_output[0].size()[-1]
        # flatten tensors to compute NLLLoss
        loss_non_terminal = base_criterion(
            n_output[0].view(-1, sz_non_terminal),
            n_target[0].view(-1)
        )

        sz_terminal = n_output[1].size()[-1]
        # flatten tensors to compute NLLLoss
        loss_terminal = base_criterion(
            n_output[1].view(-1, sz_terminal),
            n_target[1].view(-1)
        )

        return loss_non_terminal + loss_terminal

    return criterion
