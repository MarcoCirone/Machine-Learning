import numpy as np
import scipy.optimize
import numpy
from models.models import Model
from general.utils import k_fold
import matplotlib.pyplot as plt
from general.utils import *
import os


class LR(Model):
    def __init__(self, reg_term, pt):
        super().__init__()
        self.reg_term = reg_term
        self.pt = pt
        self.w = None
        self.b = None

    def set_data(self, dtr, ltr, dte, lte):
        super().set_data(dtr, ltr, dte, lte)

    def train(self):
        logreg = logreg_obj_wrap(self.dtr, self.ltr, self.reg_term, self.pt)
        x0 = numpy.zeros(self.dtr.shape[0] + 1)
        x, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg, x0=x0, approx_grad=True, factr=1.0)
        self.w = x[0:-1]
        self.b = x[-1]

    def get_scores(self):
        self.scores = np.dot(self.w.T, self.dte) + self.b
        return self.scores

    def description(self):
        return f"LR_l_{self.reg_term}_pt_{self.pt}"

    def folder(self):
        return "LR"


class QLR(LR):

    def __init__(self, reg_term, pt):
        super().__init__(reg_term, pt)

    def train(self):
        self.dtr, self.dte = feature_expansion(self.dtr, self.dte)
        super().train()

    def description(self):
        return f"QLR_l_{self.reg_term}_pt_{self.pt}_"

    def folder(self):
        return "QLR"


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


def cross_val_log_reg(d, l, prior, cfn, cfp):
    values = np.logspace(-5, 5, num=31)
    label = "$lambda$"
    e = [0.1, 0.5, 0.9]
    k = 0
    z_s = [False, True]
    for z in z_s:
        mins = [[], [], []]
        for v in values:
            reg_term = v
            model = LR(reg_term, prior)
            scores = k_fold(d, l, 5, model, seed=27, zscore=z)
            for j in range(len(e)):
                dcf_min = compute_min_dcf(scores, l, e[j], cfn, cfp)
                mins[j].append(dcf_min)
                k += 1
                print(f"Iterazione {k}: prior= {e[j]} {label}= {v} => min_dcf= {dcf_min}")
        # min_dcf_list.append(mins.copy())

        for i in range(len(mins)):
            plt.plot(values, mins[i], label=f"eff_p={e[i]}")
        plt.xscale("log")
        plt.xlabel(label)
        plt.ylabel("minDCF")
        plt.xlim([values[0], values[-1]])
        plt.legend()
        plt.show()


def cross_val_quad_log_reg(d, l, prior, cfn, cfp):
    values = np.logspace(-5, 5, num=31)
    label = "$lambda$"
    e = [0.1, 0.5, 0.9]
    k = 0
    z_s = [False, True]
    for z in z_s:
        mins = [[], [], []]
        for v in values:
            reg_term = v
            model = QLR(reg_term, prior)
            scores = k_fold(d, l, 5, model, seed=27, zscore=z)
            for j in range(len(e)):
                dcf_min = compute_min_dcf(scores, l, e[j], cfn, cfp)
                mins[j].append(dcf_min)
                k += 1
                print(f"Iterazione {k}: prior= {e[j]} {label}= {v} => min_dcf= {dcf_min}")
        # min_dcf_list.append(mins.copy())

        for i in range(len(mins)):
            plt.plot(values, mins[i], label=f"eff_p={e[i]}")
        plt.xscale("log")
        plt.xlabel(label)
        plt.ylabel("minDCF")
        plt.xlim([values[0], values[-1]])
        plt.legend()
        z_desc = ""
        if z:
            z_desc = "_zscore"
        if not os.path.exists("figures/QLR"):
            os.makedirs("figures/QLR")
        plt.savefig(f"figures/QLR/QLR{z_desc}")
        plt.show()
