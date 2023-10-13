import numpy as np
import sys
sys.path.append("../../")
from general.utils import mcol, logpdf_GAU_ND
from models.models import *

class GaussianModel(Model):
    def __init__(self):
        super().__init__()
        self.mu = []
        self.cov = []

    def mu_and_covariance(self):
        for i in range(2):
            di = self.dtr[:, self.ltr == i]  # samples for class i
            self.mu.append(di.mean(1))  # mean for class i
            center_data = di - mcol(di.mean(1))  # center dataset
            self.cov.append(np.dot(1 / (di.shape[1]) * center_data, center_data.T))  # compute covariance matrix

    def score_as_vec(self):
        self.scores = np.vstack(self.scores)
        com = np.zeros(self.score.shape[1])
        for i in range(self.score.shape[1]):
            com[i] = self.score[1][i] - self.score[0][i]
        return com

    def train(self):
        self.mu_and_covariance()

    @abstractmethod
    def get_scores(self):
        pass

class MVG(GaussianModel):
    def __init__(self, mu=None, cov=None):
        super().__init__()
        self.mu = mu if mu is not None else []
        self.cov = cov if cov is not None else []

    def get_scores(self):
        for i in range(2):
            self.scores.append(logpdf_GAU_ND(self.dte, mcol(self.mu[i]), self.cov[i]))
        self.scores = self.score_as_vec()
        return self.scores

class NaiveBayes(GaussianModel):
    def __init__(self, mu=None, cov=None):
        super().__init__()
        self.mu = mu if mu is not None else []
        self.cov = cov if cov is not None else []

    def get_scores(self):
        for i in range(2):
            self.scores.append(logpdf_GAU_ND(self.dte, mcol(self.mu[i]), self.cov[i] * np.eye(self.cov[i].shape[0])))
        self.scores = self.score_as_vec()
        return self.scores

class MVGTied(GaussianModel):
    def __init__(self, mu=None, cov=None):
        super().__init__()
        self.mu = mu if mu is not None else []
        self.cov = cov if cov is not None else []

    def compute_tied_cov(self):
        self.cov = np.zeros([self.dtr.shape[0], self.dtr.shape[0]])
        for i in range(2):
            di = self.dtr[:, self.ltr == i]  # samples for class i
            self.cov += np.dot((di - mcol(self.mu[i])), (di - mcol(self.mu[i])).T)
        self.cov /= self.dtr.shape[1]

    def get_scores(self):
        self.compute_tied_cov()
        mvg = MVG(mu=self.mu, cov=self.cov)
        self.scores = mvg.get_scores()
        return self.scores

class NBTied(GaussianModel):
    def __init__(self):
        super().__init__()

    def get_scores(self):
        mt = MVGTied(mu=self.mu, cov=self.cov)
        mt.compute_tied_cov()
        nb = NaiveBayes(mu=mt.mu, cov=mt.cov)
        self.scores = nb.get_scores()
        return self.scores



