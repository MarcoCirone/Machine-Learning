import numpy as np
import scipy.optimize
import numpy
from models.models import Model
from general.utils import k_fold
import matplotlib.pyplot as plt


class LR(Model):
    def __init__(self, dtr, ltr, dte, lte, reg_term, pt):
        super().__init__(dtr, ltr, dte, lte)
        self.reg_term = reg_term
        self.pt = pt
        self.w = None
        self.b = None

    def train(self):
        logreg = logreg_obj_wrap(self.dtr, self.ltr, self.reg_term, self.pt)
        x0 = numpy.zeros(self.dtr.shape[0] + 1)
        x, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg, x0=x0, approx_grad=True, factr=1.0)
        self.w = x[0:-1]
        self.b = x[-1]

    def get_scores(self):
        self.scores = np.dot(self.w.T, self.dte) + self.b
        return self.scores


class QLR(LR):

    def __init__(self, dtr, ltr, dte, lte, reg_term, pt):
        super().__init__(dtr, ltr, dte, lte, reg_term, pt)

    def train(self):
        self.dtr, self.dte = feature_expansion(self.dtr, self.dte)
        super().train()


def logreg_obj_wrap(dtr, ltr, reg_term, pt):
    def logreg_obj(v):
        w = v[0:-1]
        b = v[-1]
        s = 0

        for c in range(2):
            dtr_c = dtr[:, ltr == c]
            zi = 2 * c - 1
            if c == 0:
                const_class = (1 - pt) / dtr_c.shape[1]
            else:
                const_class = pt / dtr_c.shape[1]
            s += const_class * np.logaddexp(0, -zi * (numpy.dot(w.T, dtr_c) + b)).sum()

        return reg_term / 2 * (w * w).sum() + s

    return logreg_obj


def linear_log_reg(dtr, ltr, dte, p, reg_term):
    logreg = logreg_obj_wrap(dtr, ltr, reg_term, p)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg, x0=numpy.zeros(dtr.shape[0] + 1), approx_grad=True, factr=1.0)
    w = x[0:-1]
    b = x[-1]

    scores = np.dot(w.T, dte) + b
    return scores


def quad_log_reg(dtr, ltr, dte, p, reg_term):
    n_dtr, n_dte = feature_expansion(dtr, dte)
    return linear_log_reg(n_dtr, ltr, n_dte, p, reg_term)


def feature_expansion(dtr, dte):
    new_dtr = expand_matrix(dtr)
    new_dte = expand_matrix(dte)

    return new_dtr, new_dte


def expand_matrix(mat):
    dim = mat.shape[0]
    n = mat.shape[1]
    new_dim = dim ** 2 + dim
    new_mat = numpy.zeros((new_dim, n))

    for i in range(n):
        column = np.zeros((new_dim, 1))
        x = mat[:, i:i + 1]
        xx_t = numpy.dot(x, x.T)
        for j in range(dim):
            column[j * dim:j * dim + dim, :] = xx_t[:, j:j + 1]
        column[dim ** 2:dim ** 2 + dim, :] = x
        new_mat[:, i:i + 1] = column
    return new_mat
