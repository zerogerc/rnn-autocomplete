from zerogercrnn.lib.old.converter import convert_row_text_to_one_hot


class Reader:
    def get_data(self):
        pass


class RowTextDataReader(Reader):
    """Reader that provides data as a string.
    
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

        self.__data__ = self.__read_data__(file)

    def get_data(self):
        """Row string of all file.

        :return: tensor of size [N x K] where N is length of text and K is vocabulary size.
        """
        return self.__data__

    def __read_data__(self, file):
        """Reads data from file and returns row string. Fills vocabulary and all_letters."""
        with open(file=file, mode='r', encoding='ISO-8859-1') as kernel_file:
            row_data = kernel_file.read()

            for c in row_data:
                self.vocabulary.add(c)

            self.all_letters = ''.join(sorted(self.vocabulary))
            return row_data


class OneHotTextDataReader(RowTextDataReader):
    """Reader for row text data, will read text and provide data as a one-hot matrix.

    Attributes:
        vocabulary: set of all characters encountered in ``.file``.
        all_letters: string of all characters encountered in ``.file``.
                   this string is used for one hot encoding of data.
        
    Parameters:
        file (str): path to the file with input text.
    """

    def __init__(self, file):
        super(OneHotTextDataReader, self).__init__(file)
        self.__data__ = convert_row_text_to_one_hot(self.__data__, self.all_letters)

    def get_data(self):
        """One-hot encoding of input.
            
        :return: tensor of size [N x K] where N is length of text and K is vocabulary size.
        """
        return self.__data__
