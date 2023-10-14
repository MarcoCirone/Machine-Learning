from abc import abstractmethod


class Model:

    def __init__(self):
        self.dtr = None
        self.dte = None
        self.ltr = None
        self.lte = None
        self.scores = None

    def set_data(self, dtr, ltr, dte, lte):
        self.dtr = dtr
        self.dte = dte
        self.ltr = ltr
        self.lte = lte

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_scores(self):
        pass

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def folder(self):
        pass
