import os

import numpy as np
import torch
from torch.autograd import Variable

from zerogercrnn.experiments.linux.token_level.main import create_network
from zerogercrnn.lib.data.token_level import UNK_TKN
from zerogercrnn.lib.data.token_level import tokenize_file, convert_tokens
from zerogercrnn.lib.train.config import Config
from zerogercrnn.lib.utils.state import load_if_saved
from zerogercrnn.lib.visualization.html_helper import HtmlBuilder, show_html_page, string_to_html

SEQ_LEN = 10000

CONFIG_FILE = "/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/token_level/results/4Mar2018_gru/parameters.json"
MODEL_FILE = "/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/token_level/results/4Mar2018_gru/model_epoch_5"
TOKENS_FILE = "/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/token_level/results/4Mar2018_gru/tokens.txt"
SMALL_DATA_FILE = "/Users/zerogerc/Yandex.Disk.localized/shared_files/University/diploma/rnn-autocomplete/zerogercrnn/experiments/linux/data_dir/linux_kernel_mini.txt"

COLOR_GREEN = "#81C784"
COLOR_YELLOW = "#FFF59D"
COLOR_ORANGE = "#FFAB91"
COLOR_RED = "#EF9A9A"


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


def replace_newline(c):
    if c == '\n':
        return 'NL'
    else:
        return c


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

    output = network(Variable(input_tensor)).squeeze(1)  # batch_size = 1
    top_5 = torch.topk(output, k=5, dim=1, sorted=True)[1]
    top_1000 = torch.topk(output, k=100, dim=1, sorted=True)[1]

    predicted = convert(torch.max(output, 1)[1].data, tokens=tokens)
    actual = convert(target_tensor.view(-1), tokens=tokens)


    def converter(tid, token, skipped):
        """Function that converts token and it's number to html popup with prediction results."""
        skip_html = "<span>{}</span>".format(string_to_html(skipped))
        if tid >= len(actual):
            return skip_html + HtmlBuilder.get_popup_html(anchor=token, popup="Not in input")

        assert tid == 0 or actual[tid - 1] == UNK_TKN or token == actual[tid - 1]

        if tid == 0 or predicted[tid - 1] == actual[tid - 1]:
            background = COLOR_GREEN
        else:
            top_5_prev = convert(top_5[tid - 1].data, tokens=tokens)
            top_1000_prev = convert(top_1000[tid - 1].data, tokens=tokens)

            orange = False
            for t in top_1000_prev:
                if t[0] == token[0]:
                    if t == token:
                        orange = True
                    break

            if token in top_5_prev:
                background = COLOR_YELLOW
            elif orange:
                background = COLOR_ORANGE
            else:
                background = COLOR_RED

        top_5_cur = convert(top_5[tid].data, tokens=tokens)

        skip_html = "<span>{}</span>".format(string_to_html(skipped))
        popup_html = HtmlBuilder.get_popup_html(
            anchor=token,
            popup='\n'.join([replace_newline(t) for t in top_5_cur]),
            background=background
        )
        return skip_html + popup_html


    result = convert_tokens(
        path=SMALL_DATA_FILE,
        tokens_path=TOKENS_FILE,
        converter=converter
    )

    show_html_page(
        page=HtmlBuilder.get_popup_html_page(result),
        save_file=os.path.join(os.getcwd(), 'results.html')
    )
