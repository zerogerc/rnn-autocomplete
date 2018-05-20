from abc import abstractmethod


class HealthCheck:
    """Class that do some check on the model. Usually it prints some info about model at the end of epoch."""

    @abstractmethod
    def do_check(self):
        pass


class AlphaBetaSumHealthCheck(HealthCheck):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def do_check(self):
        print('Alpha: {}'.format(self.module.mult_alpha))
        print('Beta: {}'.format(self.module.mult_beta))