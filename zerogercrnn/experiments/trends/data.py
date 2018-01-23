import pandas as pd
import torch

from global_constants import ROOT_DIR
from zerogercrnn.lib.old.reader import Reader


class TrendsDataReader(Reader):
    """
    Reader for data from Google Trends. 
    It expects number of queries to be in the column called 'count'.
    Creates data_x, data_y with shapes [SEQ_LEN, DATA_LEN, INPUT_LEN/TARGET_LEN]
    """

    # number of timestamps in one input
    # one timestamp is one month so it makes sense to make input to be one year long
    SEQ_LEN = 12

    def __init__(self):
        super(TrendsDataReader, self).__init__()
        self.row = pd.read_csv(ROOT_DIR + '/data/snowboard_timeline.csv')
        self.data_x, self.data_y = TrendsDataReader.create_data(self.row)

    def get_data(self):
        return self.data_x, self.data_y

    @staticmethod
    def create_data(table):
        timeline = table['count']

        inputs = []
        targets = []
        for first in range(len(table) - TrendsDataReader.SEQ_LEN):
            # create input and target, target is input shifted by one month
            input_seq = timeline.iloc[first:first + TrendsDataReader.SEQ_LEN]
            target_seq = timeline.iloc[first + 1:first + TrendsDataReader.SEQ_LEN + 1]

            inputs.append(torch.LongTensor(input_seq.values).view(TrendsDataReader.SEQ_LEN, 1, -1))
            targets.append(torch.LongTensor(target_seq.values).view(TrendsDataReader.SEQ_LEN, 1, -1))

        # concat all inputs into the shape [SEQ_LEN, DATA_LEN, INPUT_LEN]
        # INPUT_LEN is one because it's single number
        input_tensor = torch.cat(inputs, dim=1)

        # concat all targets into the shape [SEQ_LEN, DATA_LEN, TARGET_LEN]
        # TARGET_LEN is one because it's single number
        target_tensor = torch.cat(targets, dim=1)

        return input_tensor.float(), target_tensor.float()


data_reader = TrendsDataReader()


def read_data():
    print(data_reader.data_x[:, 1, :])
    print(data_reader.data_y[:, 1, :])
