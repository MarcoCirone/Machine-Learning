import numpy as np
from general.utils import mcol, logpdf_GAU_ND
from models.models import *

class GaussianModel(Model):
    def __init__(self):
        super().__init__()
        self.mu = []
        self.cov = []

    def mu_and_covariance(self):
        self.mu = []
        self.cov = []
        for i in range(2):
            di = self.dtr[:, self.ltr == i]  # samples for class i
            self.mu.append(di.mean(1))  # mean for class i
            center_data = di - mcol(di.mean(1))  # center dataset
            self.cov.append(np.dot(1 / (di.shape[1]) * center_data, center_data.T))  # compute covariance matrix

    def score_as_vec(self):
        sc = np.vstack(self.scores)
        com = np.zeros(sc.shape[1])
        for i in range(sc.shape[1]):
            com[i] = sc[1][i] - sc[0][i]
        return com

    def train(self):
        self.mu_and_covariance()

    @abstractmethod
    def get_scores(self):
        pass

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def folder(self):
        pass

class MVG(GaussianModel):
    def __init__(self, mu=None, cov=None):
        super().__init__()
        self.mu = mu if mu is not None else []
        self.cov = cov if cov is not None else []

    def get_scores(self):
        self.scores = []
        for i in range(2):
            self.scores.append(logpdf_GAU_ND(self.dte, mcol(self.mu[i]), self.cov[i]))
        return self.score_as_vec()

    def description(self):
        return "MVG_"

    def folder(self):
        return "MVG"

class NaiveBayes(GaussianModel):
    def __init__(self, mu=None, cov=None):
        super().__init__()
        self.mu = mu if mu is not None else []
        self.cov = cov if cov is not None else []

    def get_scores(self):
        self.scores = []
        for i in range(2):
            self.scores.append(logpdf_GAU_ND(self.dte, mcol(self.mu[i]), self.cov[i] * np.eye(self.cov[i].shape[0])))
        self.scores = self.score_as_vec()
        return self.scores

    def description(self):
        return "NaiveBayes_"

    def folder(self):
        return "NaiveBayes"

class MVGTied(GaussianModel):
    def __init__(self, mu=None, cov=None):
        super().__init__()
        self.mu = mu if mu is not None else []
        self.cov = cov if cov is not None else []

    def compute_tied_cov(self):
        cov = np.zeros([self.dtr.shape[0], self.dtr.shape[0]])
        for i in range(2):
            di = self.dtr[:, self.ltr == i]  # samples for class i
            cov += np.dot((di - mcol(self.mu[i])), (di - mcol(self.mu[i])).T)
        self.cov = [cov / self.dtr.shape[1]]*2

    def get_scores(self):
        self.compute_tied_cov()
        mvg = MVG(mu=self.mu, cov=self.cov)
        mvg.set_data(self.dtr, self.ltr, self.dte, self.lte)
        self.scores = mvg.get_scores()
        return self.scores

    def description(self):
        return "Tied_"

    def folder(self):
        return "Tied"

class NBTied(GaussianModel):
    def __init__(self):
        super().__init__()

    def get_scores(self):
        mt = MVGTied(mu=self.mu, cov=self.cov)
        mt.set_data(self.dtr, self.ltr, self.dte, self.lte)
        mt.compute_tied_cov()
        nb = NaiveBayes(mu=mt.mu, cov=mt.cov)
        nb.set_data(self.dtr, self.ltr, self.dte, self.lte)
        self.scores = nb.get_scores()
        return self.scores

    def description(self):
        return "NBTied_"

    def folder(self):
        return "NBTied"



