from abc import abstractmethod


class DataGenerator:
    """General interface for generators of data for training and validation."""

    @abstractmethod
    def get_train_generator(self):
        """Provides data for one epoch of training."""
        pass

    @abstractmethod
    def get_validation_generator(self):
        """Provides data for one validation cycle."""
        pass


class DataReader:
    """General interface for readers of text files into format for DataGenerator.
    Should provide fields for train, validation, eval."""

    def __init__(self):
        self.train_data = []
        self.validation_data = []
        self.eval_data = []
        self.eval_tails = 0
