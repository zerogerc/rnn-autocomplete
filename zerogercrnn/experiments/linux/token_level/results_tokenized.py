import os

import numpy as np
import torch
from torch.autograd import Variable

from zerogercrnn.experiments.linux.token_level.main import create_network
from zerogercrnn.experiments.linux.token_level.data import read_data_mini
from zerogercrnn.experiments.linux.token_level.gru_model import GRULinuxNetwork
from zerogercrnn.experiments.linux.constants import HOME_DIR

from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.data.token_level import tokenize_file, convert_tokens
from zerogercrnn.lib.utils.state import load_if_saved
from zerogercrnn.lib.visualization.html_helper import HtmlBuilder, show_html_page, string_to_html

SEQ_LEN = 1000

CONFIG_FILE = "/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/token_level/results/27Feb2018_gru/parameters.json"
MODEL_FILE = "/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/token_level/results/27Feb2018_gru/model_epoch_5"
TOKENS_FILE = "/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/token_level/results/27Feb2018_gru/tokens.txt"
SMALL_DATA_FILE = "/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/data_dir/linux_kernel_mini.txt"


def pick_single_input(data_tensor, start, ntokens):
    return data_tensor[np.arange(start, start + SEQ_LEN), :]


def pick_single_target(data_tensor, start):  # index of target (input shifted by one)
    return data_tensor[np.arange(start + 1, start + 1 + SEQ_LEN), :]


def convert(token_positions, tokens):
    """
    Convert numbers to tokens using tokens list.
    """
    tokenized = []
    for lp in token_positions:
        tokenized.append(tokens[lp])
    return tokenized


if __name__ == '__main__':
    config = Config()
    config.read_from_file(CONFIG_FILE)

    data, tokens = tokenize_file(
        path=SMALL_DATA_FILE,
        tokens_path=TOKENS_FILE
    )
    data = data.unsqueeze(1)

    input_tensor = pick_single_input(data, start=0, ntokens=len(tokens))
    target_tensor = pick_single_target(data, start=0)

    network = create_network(config, len(tokens))
    load_if_saved(network, MODEL_FILE)

    output = network(Variable(input_tensor))

    predicted = convert(torch.max(output.squeeze(1), 1)[1].data, tokens=tokens)
    actual = convert(target_tensor.view(-1), tokens=tokens)

    result = convert_tokens(
        path=SMALL_DATA_FILE,
        tokens_path=TOKENS_FILE,
        converter=lambda tid, token, skipped:
        "<span>{}</span>".format(string_to_html(skipped)) +
        HtmlBuilder.get_popup_html(token, predicted[min(tid, len(predicted) - 1)])
    )

    show_html_page(
        page=HtmlBuilder.get_popup_html_page(result[:100000]),
        save_file=os.path.join(os.getcwd(), 'results.html')
    )
