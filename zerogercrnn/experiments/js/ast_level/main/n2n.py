import torch.nn as nn

from zerogercrnn.experiments.js.ast_level.main.common import get_optimizers_and_schedulers, load_if_saved_from_config
from zerogercrnn.experiments.js.ast_level.model.n2n import NTModel, RecurrentCore
from zerogercrnn.experiments.js.ast_level.routine.n2n import NTASTRoutine
from zerogercrnn.lib.train.run import TrainEpochRunner


def nt_run_training(cfg, title, cuda, data_generator):
    criterion = get_criterion(cuda=cuda)

    model = nt_get_model(cfg, cuda)
    # load_if_saved_from_config(cfg, model)
    optimizers, schedulers = get_optimizers_and_schedulers(cfg, model)

    train_routine = NTASTRoutine(
        model=model,
        batch_size=cfg.batch_size,
        criterion=criterion,
        optimizers=optimizers,
        cuda=cuda
    )

    validation_routine = NTASTRoutine(
        model=model,
        batch_size=cfg.batch_size,
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
        save_dir=cfg.model_save_dir
    )

    runner.run(number_of_epochs=cfg.epochs)


def nt_get_model(cfg, cuda):
    # Model
    # recurrent_core = RecurrentCore(
    #     input_size=cfg.embedding_size,
    #     hidden_size=cfg.hidden_size,
    #     num_layers=cfg.num_layers,
    #     dropout=cfg.dropout,
    #     model_type='gru'
    # )
    recurrent_core = RecurrentCore(
        input_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        model_type='lstm'
    )
    if cuda:
        recurrent_core = recurrent_core.cuda()

    model = NTModel(
        non_terminal_vocab_size=cfg.non_terminals_count,
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
        sz_non_terminal = n_output.size()[-1]
        # flatten tensors to compute NLLLoss
        loss_non_terminal = base_criterion(
            n_output.view(-1, sz_non_terminal),
            n_target.view(-1)
        )

        return loss_non_terminal

    return criterion
