
import numpy as np
import  torch

def split_rnn_datasets(input_tensor, target_tensor, validation_percentage=0.2, test_percentage=0.2, shuffle=True):
    """
    Splits input and target tensors into train/validation/test. 
    Validation dataset size will be equal to len(input) * validation_percentage.
    Test dataset size will be equal to len(input) * test_percentage.
    All other entries will be train dataset.
    
    :param input_tensor: tensor of the size (seq_len, data_size, input_size)
    :param target_tensor: (seq_len, data_size)
    :param validation_percentage: percentage of validation data
    :param test_percentage: percentage of test data
    :return: 6 datasets : input/target for train, validation, test
    """
    n = input_tensor.size()[1]

    train_indexes, validation_indexes, test_indexes = \
        get_split_indexes(n, validation_percentage, test_percentage, shuffle)

    # creating datasets
    validation_x = input_tensor[:, validation_indexes, :]
    validation_y = target_tensor[:, validation_indexes]

    test_x = input_tensor[:, test_indexes, :]
    test_y = target_tensor[:, test_indexes]

    train_x = input_tensor[:, train_indexes, :]
    train_y = target_tensor[:, train_indexes]

    return train_x, train_y, validation_x, validation_y, test_x, test_y


def split_data(data_tensor, validation_percentage=0.2, test_percentage=0.2, shuffle=True):
    """
    Splits 2d data tensor

    :param data_tensor: tensor of the size (data_size, input_size)
    :param validation_percentage: percentage of validation data
    :param test_percentage: percentage of test data
    :return: 3 datasets : train, validation, test
    """
    train_indexes, validation_indexes, test_indexes = \
        get_split_indexes(len(data_tensor), validation_percentage, test_percentage, shuffle)

    # creating datasets
    train = data_tensor[train_indexes, :]
    validation = data_tensor[test_indexes, :]
    test = data_tensor[test_indexes, :]

    return train, validation, test


def get_split_indexes(length, validation_percentage=0.2, test_percentage=0.2, shuffle=True):
    # create indexes sequence and shuffle them if needed
    indexes = np.arange(0, length)
    if shuffle:
        np.random.shuffle(indexes)

    # calculating sizes of datasets
    validation_size = np.floor(validation_percentage * length).astype(int)
    test_size = np.floor(test_percentage * length).astype(int)

    # getting indexes of datasets
    test_indexes = indexes[-validation_size:]
    validation_indexes = indexes[-(validation_size + test_size):-validation_size]
    train_indexes = indexes[:-(validation_size + test_size)]

    return train_indexes, validation_indexes, test_indexes

if __name__ == '__main__':
    print(split_data(torch.eye(100), shuffle=False))