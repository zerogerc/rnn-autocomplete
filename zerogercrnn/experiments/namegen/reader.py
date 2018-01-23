import numpy as np
import torch
from lib.old.reader import Reader

from zerogercrnn.lib.utils.file import find_files, read_lines, all_letters, n_letters

max_name_length = 20


class NamesDataReader(Reader):
    def __init__(self):
        super(NamesDataReader, self).__init__()
        self.category_lines = {}
        self.all_categories = []
        self.n_categories = 0
        self.__read_data__()

    def get_data(self):
        """
        :return: **input_tensor** of size (seq_len, data_len, input_size), 
                 **target_tensor** of size (seq_len, data_len)
        """
        category_data = torch.zeros(0)
        input_data = torch.zeros(0)
        target_data = torch.zeros(0).long()

        for category in self.category_lines:
            category_tensor = self.category_tensor(category)

            lines = self.category_lines[category]
            category_inputs = [self.input_tensor(line) for line in lines]
            category_targets = [self.target_tensor(line) for line in lines]

            cur_category = []
            cur_inputs = []
            cur_targets = []
            for i in range(len(category_inputs)):
                cur_category.append(category_tensor)
                cur_inputs.append(category_inputs[i])
                cur_targets.append(category_targets[i].view(-1, 1))

            category_data = torch.cat([category_data, *cur_category], dim=0)
            input_data = torch.cat([input_data, *cur_inputs], dim=1)
            target_data = torch.cat([target_data, *cur_targets], dim=1)

        isz = input_data.size()
        category_data = category_data.expand(input_data.size()[0], category_data.size()[0], category_data.size()[1])
        input_data = torch.cat((category_data, input_data), 2)

        return input_data, target_data

    # Converts output of the network to a input-like tensor with category for a timestamp
    #
    # Works only with batch_size = 1, as it's used only during prediction
    # Returns None if output is ES
    def output_to_input_with_category(self, category_tensor, output):
        cur_input = output

        topv, topi = cur_input.data.topk(1)
        topi = topi[0][0]

        if topi == n_letters - 1:
            return None
        else:
            cur_input = self.input_tensor(all_letters[topi])[0]  # create one-hot tensor of letter with max weight

        return torch.cat((category_tensor, cur_input), 1)  # concat category and letter tensor

    # Build the category_lines dictionary, a list of lines per category
    # returns list of categories, category_lines
    def __read_data__(self):
        for filename in find_files('data/names/*.txt'):
            category = filename.split('/')[-1].split('.')[0]
            self.all_categories.append(category)
            lines = read_lines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)

    # One-hot vector for category
    def category_tensor(self, category):
        li = self.all_categories.index(category)
        tensor = torch.zeros(1, self.n_categories)
        tensor[0][li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    @classmethod
    def input_tensor(cls, line):
        tensor = torch.zeros(max_name_length, 1, n_letters)
        for li in range(max_name_length):
            if li < len(line):
                letter = line[li]
                tensor[li][0][all_letters.find(letter)] = 1
            else:
                # append EOS to make all inputs to be the same length
                tensor[li][0][n_letters - 1] = 1

        return tensor

    # LongTensor of second letter to end (EOS) for target
    @classmethod
    def target_tensor(cls, line):
        letter_indexes = []
        for li in range(1, max_name_length):
            if li < len(line):
                letter_indexes.append(all_letters.find(line[li]))
            else:
                # append EOS to make all inputs to be the same length
                letter_indexes.append(n_letters - 1)
        # last character should be EOS
        letter_indexes.append(n_letters - 1)  # EOS
        return torch.from_numpy(np.array(letter_indexes, dtype=np.int64))


names_data_reader = NamesDataReader()
