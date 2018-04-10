import torch.nn as nn

from zerogercrnn.experiments.js.ast_level.main.common import get_optimizers_and_schedulers
from zerogercrnn.experiments.js.ast_level.model.n2n_sum_attention_sequential import NTSumlAttentionModelSequential
from zerogercrnn.experiments.js.ast_level.routine.n2n_sequential import N2NSequential
from zerogercrnn.lib.train.run import TrainEpochRunner


def nt_seq_sum_attention_run_training(cfg, title, cuda, data_generator, model_save_dir):
    criterion = nn.NLLLoss()

    if cuda:
        criterion = criterion.cuda()

    model = NTSumlAttentionModelSequential(
        non_terminal_vocab_size=cfg.non_terminals_count,
        embedding_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        seq_len=cfg.seq_len
    )

    if cuda:
        model = model.cuda()

    optimizers, schedulers = get_optimizers_and_schedulers(cfg, model)

    train_routine = N2NSequential(
        model=model,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        criterion=criterion,
        optimizers=optimizers,
        cuda=cuda
    )

    validation_routine = N2NSequential(
        model=model,
        seq_len=cfg.seq_len,
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
