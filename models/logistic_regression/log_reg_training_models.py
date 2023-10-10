import numpy as np
import scipy.optimize
import numpy
from general.utils import k_fold
import matplotlib.pyplot as plt


def logreg_obj_wrap(dtr, ltr, reg_term, p):
    def logreg_obj(v):
        w = v[0:-1]
        b = v[-1]
        s = 0

        for c in range(2):
            dtr_c = dtr[:, ltr == c]
            zi = 2 * c - 1
            if c == 0:
                const_class = (1 - p) / dtr_c.shape[1]
            else:
                const_class = p / dtr_c.shape[1]
            s += const_class * np.logaddexp(0, -zi * (numpy.dot(w.T, dtr_c) + b)).sum()

        return reg_term / 2 * (w * w).sum() + s

    return logreg_obj

# def logreg_obj_wrap(dtr, ltr, reg_term, p):
#     z = 2 * ltr - 1
#     priors = [1-p, p]
#
#     def logreg_obj(v):
#         w, b = v[0:-1], v[-1]
#         s = 0
#         const = (reg_term / 2) * (np.dot(w, w.T))
#         for i in range(np.unique(ltr).size):
#             const_class = (priors[i] / dtr[:, ltr == i].shape[1])
#             s += const_class * np.logaddexp(0, -z[ltr == i] * (np.dot(w.T, dtr[:, ltr == i]) + b)).sum()
#
#         return const + s
#
#     return logreg_obj


def linear_log_reg(dtr, ltr, dte, p, reg_term):
    logreg = logreg_obj_wrap(dtr, ltr, reg_term, p)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg, x0=numpy.zeros(dtr.shape[0] + 1), approx_grad=True, factr=1.0)
    w = x[0:-1]
    b = x[-1]

    scores = np.dot(w.T, dte) + b
    return scores


# def feature_expansion(dtr, dte):
#     n_train = dtr.shape[1]
#     n_eval = dte.shape[1]
#     n_f = dtr.shape[0] ** 2 + dtr.shape[0]
#     quad_dtr = np.zeros((n_f, n_train))
#     quad_dte = np.zeros((n_f, n_eval))
#
#     for i in range(n_train):
#         quad_dtr[:, i:i + 1] = stack(dtr[:, i:i + 1])
#     for i in range(n_eval):
#         quad_dte[:, i:i + 1] = stack(dte[:, i:i + 1])
#
#     return quad_dtr, quad_dte
#
#
# def stack(array):
#     n_f = array.shape[0]
#     xx_t = np.dot(array, array.T)
#     column = np.zeros((n_f ** 2 + n_f, 1))
#     for i in range(n_f):
#         column[i * n_f:i * n_f + n_f, :] = xx_t[:, i:i + 1]
#     column[n_f ** 2: n_f ** 2 + n_f, :] = array
#    return column


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
        x = mat[:, i:i+1]
        xx_t = numpy.dot(x, x.T)
        for j in range(dim):
            column[j * dim:j * dim + dim, :] = xx_t[:, j:j+1]
        column[dim**2:dim**2 + dim, :] = x
        new_mat[:, i:i + 1] = column
    return new_mat

def plot_min_dcfs(dtr, ltr, cfn, cfp, pt):
    lam_values = np.logspace(-5, 5, num=51)
    min_dcf_list = []
    e = [0.5, 0.1, 0.9]
    k = 0
    for eff_p in e:
        mins = []
        for lam in np.logspace(-5, 5, num=51):
            mins.append(k_fold(dtr, ltr, 5, linear_log_reg, eff_p, cfn, cfp, seed=27, pt=pt, reg_term=lam))
            k += 1
            print(f"Iterazione {k}")
        min_dcf_list.append(mins.copy())
    for i in range(len(min_dcf_list)):
        plt.plot(lam_values, min_dcf_list[i], label=f"eff_p={e[i]}")
    plt.xscale("log")
    plt.xlabel("$lambda$")
    plt.ylabel("minDCF")
    plt.xlim([lam_values[0], lam_values[-1]])
    plt.legend()
    plt.show()
