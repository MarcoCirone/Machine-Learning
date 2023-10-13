from abc import abstractmethod


class Model:

    def __init__(self, dtr, ltr, dte, lte):
        self.dtr = dtr
        self.dte = dte
        self.ltr = ltr
        self.lte = lte
        self.scores = None

    def set_new_data(self, dtr, dte):
        self.dtr = dtr
        self.dte = dte

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_scores(self):
        pass
