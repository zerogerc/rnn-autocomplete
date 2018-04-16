from zerogercrnn.experiments.ast_level.data import DataReader
from zerogercrnn.lib.core import RecurrentCore
from zerogercrnn.lib.utils.state import load_if_saved, load_cuda_on_cpu


def create_model(cuda, cfg, model_path, constructor):
    model = constructor()

    if cuda:
        model = model.cuda()

    # Model is always trained with cuda so if we evaluate model on non cuda device
    # we should call another function for loading
    if cuda:
        load_if_saved(model, model_path)
    else:
        load_cuda_on_cpu(model, model_path)

    return model


def create_recurrent_core(cuda, cfg, model_type):
    recurrent_core = RecurrentCore(
        input_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        model_type=model_type
    )

    if cuda:
        recurrent_core = recurrent_core.cuda()

    return recurrent_core


def create_eval_reader(cuda, cfg, eval_file, eval_limit):
    reader = DataReader(
        file_training=None,
        file_eval=eval_file,
        encoding=cfg.encoding,
        limit_train=None,
        limit_eval=eval_limit,
        cuda=cuda
    )

    return reader
