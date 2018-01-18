import torch


def convert_row_text_to_one_hot(row_text, all_letters):
    """Converts row text to one-hot float tensor.
    **See** test/data/converter_test.py for usages.
    
    :param row_text: text to convert.
    :param all_letters: concatenated string of alphabet.
    :return: one-hot tensor of size [N x K] where N is length of text and K is vocabulary size.
    """
    positions = convert_row_text_to_index_tensor(row_text, all_letters)
    one_hot = torch.zeros(len(row_text), len(all_letters))
    one_hot.scatter_(1, positions.view(len(row_text), 1), 1.)
    return one_hot


def convert_row_text_to_index_tensor(row_text, all_letters):
    """Converts row text to long tensor with indexes.
    **See** test/data/converter_test.py for usages.
    
    :param row_text: text to convert.
    :param all_letters: concatenated string of alphabet.
    :return: tensor of size [N] where N is length of text.
    """
    return torch.LongTensor(list(map(all_letters.find, row_text)))
