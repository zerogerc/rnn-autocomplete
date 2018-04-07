import torch.nn as nn

from zerogercrnn.experiments.js.ast_level.main.common import get_optimizers_and_schedulers, load_if_saved_from_config
from zerogercrnn.experiments.js.ast_level.model.n2n_attention import NTTailAttentionModel, NTTailAttentionModel2Softmax, \
    RecurrentCore
from zerogercrnn.experiments.js.ast_level.routine.n2n_attention import NTTailAttentionASTRoutine
from zerogercrnn.lib.train.run import TrainEpochRunner


def nt_at_run_training(cfg, title, cuda, data_generator, model_save_dir):
    criterion = nn.NLLLoss()

    if cuda:
        criterion = criterion.cuda()

    model = nt_get_model(cfg, cuda)
    optimizers, schedulers = get_optimizers_and_schedulers(cfg, model)

    train_routine = NTTailAttentionASTRoutine(
        model=model,
        batch_size=cfg.batch_size,
        criterion=criterion,
        optimizers=optimizers,
        cuda=cuda
    )

    validation_routine = NTTailAttentionASTRoutine(
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
        save_dir=model_save_dir
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

    model = NTTailAttentionModel2Softmax(
        non_terminal_vocab_size=cfg.non_terminals_count,
        embedding_size=cfg.embedding_size,
        recurrent_layer=recurrent_core
    )

    if cuda:
        model = model.cuda()

    return model
