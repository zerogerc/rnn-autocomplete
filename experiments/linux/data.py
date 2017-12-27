import torch

from constants import ROOT_DIR
from lib.data.reader import Reader
from lib.data.text_reader import RowTextDataReader


class LinuxKernelDataReader(Reader):
    """
    Reader for data of Linux Kernel. 
    Creates data_x, data_y with shapes [SEQ_LEN, DATA_LEN, INPUT_LEN/TARGET_LEN]
    """

    def __init__(self):
        super(LinuxKernelDataReader, self).__init__()
        self.row_text_reader = RowTextDataReader(ROOT_DIR + '/data/linux_kernel_mini.txt')
        self.data_tensor = self.row_text_reader.get_data()


data_reader = LinuxKernelDataReader()