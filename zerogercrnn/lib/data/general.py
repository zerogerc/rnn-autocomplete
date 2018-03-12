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
