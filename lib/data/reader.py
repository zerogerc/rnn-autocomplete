import torch


class Reader:
    def get_data(self):
        pass


class RowTextDataReader(Reader):
    """Reader for row text data, will read text and provide data as a one-hot matrix.

    Attributes:
        vocabulary: set of all characters encountered in ``.file``.
        all_letters: string of all characters encountered in ``.file``.
                   this string is used for one hot encoding of data.
        
    Parameters:
        file (str): path to the file with input text.
    """

    def __init__(self, file):
        super(RowTextDataReader, self).__init__()
        self.vocabulary = set()
        self.all_letters = ""

        self.__one_hot__ = self.__read_data_to_tensor__(file)

    def get_data(self):
        """One-hot encoding of input.
            
        :return: tensor of size [N x K] where N is length of text and K is vocabulary size.
        """
        return self.__one_hot__

    def __read_data_to_tensor__(self, file):
        """Reads data from file and returns one-hot matrix of the size [N x K]
        where N is the number of characters in input and K is the vocabulary size.
        """
        with open(file=file, mode='r', encoding='ISO-8859-1') as kernel_file:
            row_data = kernel_file.read()

            for c in row_data:
                self.vocabulary.add(c)

            self.all_letters = ''.join(sorted(self.vocabulary))

            positions = torch.LongTensor(list(map(self.all_letters.find, row_data)))
            one_hot = torch.FloatTensor(len(row_data), len(self.vocabulary))
            one_hot.scatter_(1, positions.view(len(row_data), 1), 1.)
            return one_hot