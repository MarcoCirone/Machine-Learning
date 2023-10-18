import numpy as np
import scipy.optimize as sp
import sys
sys.path.append("../../")
from general.utils import mcol, mrow, k_fold
from itertools import repeat
from models.models import *


def dual_obj_wrap(h):
    def dual_obj(alpha):
        grad = np.dot(h, alpha) - np.ones(h.shape[1])
        return 0.5 * np.dot(alpha.T, np.dot(h, alpha)) - np.dot(alpha, np.ones(h.shape[1])), grad

    return dual_obj

class SvmModel(Model):
    def __init__(self, c=None, k=1, pt=None, preprocess="Raw"):
        super().__init__()
        self.k = k
        self.c = c
        self.pt = pt
        self.preprocess = preprocess

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

    def set_values(self, c, pt):
        self.c = c
        self.pt = pt

    def compute_c1_c0(self):
        p_emp = (self.ltr == 1).sum() / self.ltr.shape[0]
        c1 = self.c * (self.pt / p_emp)
        c0 = self.c * ((1 - self.pt) / (1 - p_emp))
        return c1, c0

    def compute_svm_steps(self, g_or_kernel):
        z = 2 * self.ltr - 1
        h = mcol(z) * mrow(z) * g_or_kernel
        if self.pt is None:
            bound = list(repeat((0, self.c), self.ltr.shape[0]))
        else:
            c1, c0 = self.compute_c1_c0()
            bound = np.zeros((self.ltr.shape[0]))
            bound[self.ltr == 1] = c1
            bound[self.ltr == 0] = c0
            bound = list(zip(np.zeros(self.ltr.shape[0]), bound))
        dual_obj = dual_obj_wrap(h)
        alpha, _, _ = sp.fmin_l_bfgs_b(dual_obj, np.zeros(self.ltr.shape[0]), bounds=bound, factr=1.0)
        return alpha, z

class LinearSvm(SvmModel):
    def __init__(self, c=None, k=1, pt=None, preprocess="Raw"):
        super().__init__(c, k, pt, preprocess)
        self.w = None

    def train(self):
        dtr_ext = np.vstack([self.dtr, np.ones(self.dtr.shape[1]) * self.k])
        g = np.dot(dtr_ext.T, dtr_ext)
        alpha, z = self.compute_svm_steps(g)
        self.w = np.sum((alpha * z).reshape(1, self.dtr.shape[1]) * dtr_ext, axis=1)

    def get_scores(self):
        dte_ext = np.vstack([self.dte, np.ones(self.dte.shape[1]) * self.k])
        self.scores = np.dot(self.w.T, dte_ext)
        return self.scores

    def description(self):
        return f"Linear_SVM_{self.c}_k_{self.k}_pt_{self.pt}_"

    def folder(self):
        return f"Linear_SVM/{self.preprocess}"

class KernelSvm(SvmModel):
    def __init__(self, c, k, pt, preprocess="Raw"):
        super().__init__(c, k, pt, preprocess)
        self.kernel = None
        self.alpha_z = None

    def train(self):
        self.kernel = self.compute_kernel(self.dtr, self.dtr)
        alpha, z = self.compute_svm_steps(self.kernel)
        self.alpha_z = (alpha*z).reshape(1, self.dtr.shape[1])

    def get_scores(self):
        self.scores = np.sum(np.dot(self.alpha_z, self.compute_kernel(self.dtr, self.dte)), axis=0)
        return self.scores

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def compute_kernel(self, d1, d2):
        pass

    @abstractmethod
    def folder(self):
        pass

class PolSvm(KernelSvm):
    def __init__(self, constant, dimension, c=None, k=1, pt=None, preprocess="Raw"):
        super().__init__(c, k, pt, preprocess)
        self.constant = constant
        self.dimension = dimension

    def compute_kernel(self, d1, d2):
        return ((np.dot(d1.T, d2) + self.constant) ** self.dimension) + self.k ** 2

    def description(self):
        return f"Polinomial_SVM_{self.c}_k_{self.k}_pt_{self.pt}_c_{self.constant}_d_{self.dimension}"

    def folder(self):
        return f"Polinomial_SVM/{self.preprocess}"

class RbfSvm(KernelSvm):

    def __init__(self, gamma, c=None, k=1, pt=None, preprocess="Raw"):
        super().__init__(c, k, pt, preprocess)
        self.gamma = gamma

    def compute_kernel(self, d1, d2):
        rbf_kernel = np.zeros((d1.shape[1], d2.shape[1]))
        for i in range(d1.shape[1]):
            for j in range(d2.shape[1]):
                rbf_kernel[i, j] = np.exp(-self.gamma * (np.linalg.norm(d1[:, i] - d2[:, j]) ** 2)) + self.k ** 2
        return rbf_kernel

    def description(self):
        return f"Rbf_SVM_{self.c}_k_{self.k}_pt_{self.pt}_gamma_{self.gamma}"

    def folder(self):
        return f"Rbf_SVM/{self.preprocess}"
