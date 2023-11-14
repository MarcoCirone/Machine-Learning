import numpy as np
import math
import scipy.special as sp
from general.utils import logpdf_GAU_ND, mrow, mcol
from models.models import *


def comp_params(res, x, comp_cov, psi):
    new_gmm = []
    for g in range(res.shape[0]):
        gamma = res[g, :]
        z = gamma.sum()
        f = (mrow(gamma) * x).sum(1)
        s = np.dot(x, (mrow(gamma) * x).T)
        w = z / x.shape[1]
        mu = mcol(f / z)
        cov = s / z - np.dot(mu, mu.T)
        cov = comp_cov(cov)
        u, s, _ = np.linalg.svd(cov)
        s[s < psi] = psi
        cov = np.dot(u, mcol(s) * u.T)
        new_gmm.append((w, mu, cov))
    return new_gmm

def comp_params_tied(res, x, comp_cov, psi):
    w = np.zeros(res.shape[0])
    mu = np.zeros([res.shape[0], x.shape[0]])
    cov = np.zeros([x.shape[0], x.shape[0]])
    new_gmm = []
    for g in range(res.shape[0]):
        gamma = res[g, :]
        z = gamma.sum()
        f = (mrow(gamma) * x).sum(1)
        s = np.dot(x, (mrow(gamma) * x).T)
        w[g] = z / x.shape[1]
        mu[g] = f/z
        mt = mcol(mu[g])
        cov_new = s/z - np.dot(mt, mt.T)
        cov_new = comp_cov(cov_new)
        cov += z * cov_new
    cov /= x.shape[1]
    u, s, _ = np.linalg.svd(cov)
    s[s < psi] = psi
    cov = np.dot(u, mcol(s) * u.T)

    for i in range(res.shape[0]):
        new_gmm.append((w[i], mcol(mu[i]), cov))
    return new_gmm

def logpdf_GMM(x, gmm):
    s = np.zeros([len(gmm), x.shape[1]])
    for i in range(len(gmm)):
        s[i, :] = logpdf_GAU_ND(x, gmm[i][1], gmm[i][2])+np.log(gmm[i][0])      # gmm[i]=[w, mu, cov]
    logdens = sp.logsumexp(s, axis=0)
    return s, mrow(logdens)

class GmmModel(Model):
    def __init__(self, g_num=None, preprocess="raw"):
        super().__init__()
        self.alpha = 0.1
        self.psi = 0.01
        self.stop = 10**(-6)
        self.preprocess = preprocess
        self.g_num = g_num
        self.gmm = []

    def set_values(self, g_num):
        self.g_num = g_num

    def set_preprocess(self, preprocess):
        self.preprocess = preprocess

    def EM_algorithm(self, x, gmm):
        s, logdens = logpdf_GMM(x, gmm)
        old_log = logdens.sum() / x.shape[1]
        while True:
            # E step
            res = np.exp(s - logdens)
            # M step
            new_gmm = self.function_params(res, x)
            new_s, new_logdens = logpdf_GMM(x, new_gmm)
            new_log = new_logdens.sum() / x.shape[1]
            if new_log - old_log < self.stop:
                break
            else:
                logdens = new_logdens
                old_log = new_log
                s = new_s
                continue
        return new_gmm

    def LBG_algorithm(self, x):
        mu = mcol(x.mean(axis=1))
        cov = 1 / (x.shape[1]) * np.dot(x - mu, (x - mu).T)

        u, s, _ = np.linalg.svd(self.compute_cov(cov))
        s[s < self.psi] = self.psi
        cov = np.dot(u, mcol(s) * u.T)

        gmm = [(1.0, mu, cov)]

        while len(gmm) < self.g_num:
            gmm_1 = []
            for i in range(len(gmm)):
                u, s, vh = np.linalg.svd(gmm[i][2])
                d = u[:, 0:1] * s[0] ** 0.5 * self.alpha
                w = gmm[i][0] / 2
                mu1 = gmm[i][1] + d
                mu2 = mu1 - 2 * d
                gmm_1.append((w, mu1, gmm[i][2]))
                gmm_1.append((w, mu2, gmm[i][2]))
            gmm = self.EM_algorithm(x, gmm_1)
        return gmm

    def train(self):
        self.gmm = []
        for i in range(max(self.ltr)+1):
            x = self.dtr[:, self.ltr == i]
            self.gmm.append(self.LBG_algorithm(x))

    def get_scores(self):
        scores = np.zeros(self.dte.shape[1])
        for j in range(self.dte.shape[1]):
            tmp_score = []
            for gmm in self.gmm:
                _, logdens = logpdf_GMM(mcol(self.dte[:, j]), gmm)
                tmp_score.append(logdens)
            scores[j] = tmp_score[1] - tmp_score[0]
        self.scores=scores
        return self.scores

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def folder(self):
        pass

class GMM(GmmModel):
    def __init__(self, g_num=None, preprocess="raw"):
        super().__init__(g_num, preprocess=preprocess)

    def description(self):
        return f"GMM_{self.g_num}_"

    def folder(self):
        return f"GMM/{self.preprocess}"

    @staticmethod
    def compute_cov(cov):
        return cov

    def function_params(self, res, x):
        return comp_params(res, x, self.compute_cov, self.psi)

class GMMTied(GmmModel):
    def __init__(self, g_num=None, preprocess="raw"):
        super().__init__(g_num, preprocess=preprocess)

    def description(self):
        return f"Tied_GMM_{self.g_num}_"

    def folder(self):
        return f"Tied_GMM_/{self.preprocess}"

    @staticmethod
    def compute_cov(cov):
        return cov

    def function_params(self, res, x):
        return comp_params_tied(res, x, self.compute_cov, self.psi)

class GMMDiag(GmmModel):
    def __init__(self, g_num=None, preprocess="raw"):
        super().__init__(g_num, preprocess=preprocess)

    def description(self):
        return f"Diag_GMM_{self.g_num}_"

    def folder(self):
        return f"Diag_GMM_/{self.preprocess}"

    @staticmethod
    def compute_cov(cov):
        return cov*np.eye(cov.shape[0])

    def function_params(self, res, x):
        return comp_params(res, x, self.compute_cov, self.psi)

class GMMTiedDiag(GmmModel):
    def __init__(self, g_num=None, preprocess="raw"):
        super().__init__(g_num, preprocess=preprocess)

    def description(self):
        return f"TiedDiag_GMM_{self.g_num}_"

    def folder(self):
        return f"TiedDiag_GMM_/{self.preprocess}"

    @staticmethod
    def compute_cov(cov):
        return cov*np.eye(cov.shape[0])

    def function_params(self, res, x):
        return comp_params_tied(res, x, self.compute_cov, self.psi)




