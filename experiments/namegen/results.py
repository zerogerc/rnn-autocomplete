import torch
from torch import autograd

from lib.utils.file import n_letters, all_letters

from experiments.namegen.utils import max_name_length
from experiments.namegen.reader import names_data_reader


def sample_one(network, category, start_line):
    """Sample from a category and starting letter.
    """
    category_tensor = autograd.Variable(names_data_reader.category_tensor(category))
    input_tensor = autograd.Variable(names_data_reader.input_tensor(start_line))

    isz = input_tensor.size()
    copied_category = category_tensor.expand(isz[0], isz[1], len(names_data_reader.all_categories))
    input_tensor = torch.cat((copied_category, input_tensor), 2)

    network.zero_grad()
    output = network.predict(input_tensor)

    output_name = start_line
    for i in range(len(start_line), max_name_length):
        if i >= len(output.data):
            break
        topv, topi = output.data[i].topk(1)
        topi = topi[0][0]

        if topi == n_letters - 1:
            break
        else:
            letter = all_letters[topi]
            output_name += letter
    return output_name


def samples(network, category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample_one(network, category, start_letter))
    print('\n')
