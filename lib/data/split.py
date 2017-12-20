
import numpy as np


def split_datasets(input_tensor, target_tensor, validation_percentage=0.2, test_percentage=0.2):
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

    # create indexes sequence and shuffle them
    indexes = np.arange(0, n)
    np.random.shuffle(indexes)

    # calculating sizes of datasets
    validation_size = np.floor(validation_percentage * n).astype(int)
    test_size = np.floor(test_percentage * n).astype(int)
    train_size = n - validation_size - test_size

    # getting indexes of datasets
    validation_indexes = indexes[:validation_size]
    test_indexes = indexes[validation_size:validation_size + test_size]
    train_indexes = indexes[validation_size + test_size:]

    # creating datasets
    validation_x = input_tensor[:, validation_indexes, :]
    validation_y = target_tensor[:, validation_indexes]

    test_x = input_tensor[:, test_indexes, :]
    test_y = target_tensor[:, test_indexes]

    train_x = input_tensor[:, train_indexes, :]
    train_y = target_tensor[:, train_indexes]

    return train_x, train_y, validation_x, validation_y, test_x, test_y
