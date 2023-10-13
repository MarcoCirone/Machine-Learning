import numpy as np
import scipy.optimize as sp
import sys
sys.path.append("../../")
from general.utils import mcol, mrow, k_fold
from itertools import repeat

# linear SVM
def score_mat(dtr, ltr, dte, c, p, k=1):
    dtr_ext = np.vstack([dtr, np.ones(dtr.shape[1])*k])
    dte_ext = np.vstack([dte, np.ones(dte.shape[1])*k])
    g = np.dot(dtr_ext.T, dtr_ext)
    alpha, z = compute_svm_steps(ltr, g, p, c)
    w = np.sum((alpha*z).reshape(1, dtr.shape[1])*dtr_ext, axis=1)
    return np.dot(w.T, dte_ext)


# quadratic SVM
def score_mat_kernel(dtr, ltr, dte, c, p, k=1, rbf=None):
    kernel = comp_kernel(dtr, dtr, k, rbf=rbf)
    alpha, z = compute_svm_steps(ltr, kernel, p, c)
    scores = np.sum(np.dot((alpha*z).reshape(1, dtr.shape[1]), comp_kernel(dtr, dte, k, rbf=rbf)), axis=0)
    # scores = (mcol(alpha) * mcol(z) * comp_kernel(dtr, dte, k, rbf=rbf)).sum(0)
    return scores


def comp_kernel(d1, d2, k, rbf=None, c=1, d=2):
    if rbf is None:
        return ((np.dot(d1.T, d2) + c) ** d) + k**2
    rbf_kernel = np.zeros((d1.shape[1], d2.shape[1]))
    for i in range(d1.shape[1]):
        for j in range(d2.shape[1]):
            rbf_kernel[i, j] = np.exp(-rbf * (np.linalg.norm(d1[:, i] - d2[:, j]) ** 2)) + k**2
    return rbf_kernel

# functions for both linar and quadratic svm
def compute_svm_steps(ltr, g_or_kernel, p, c):
    z = 2*ltr-1
    h = mcol(z) * mrow(z) * g_or_kernel
    if p is None:
        bound = list(repeat((0, c), ltr.shape[0]))
    else:
        c1, c0 = compute_c1_c0(ltr, p, c)
        bound = np.zeros((ltr.shape[0]))
        bound[ltr == 1] = c1
        bound[ltr == 0] = c0
        bound = list(zip(np.zeros(ltr.shape[0]), bound))
    dual_obj = dual_obj_wrap(h)
    alpha, _, _ = sp.fmin_l_bfgs_b(dual_obj, np.zeros(ltr.shape[0]), bounds=bound, factr=1.0)
    return alpha, z
def compute_c1_c0(ltr, p, c):
    p_emp = (ltr == 1).sum() / ltr.shape[0]
    c1 = c * (p / p_emp)
    c0 = c * ((1 - p) / (1 - p_emp))
    return c1, c0

def dual_obj_wrap(h):

    def dual_obj(alpha):
        grad = np.dot(h, alpha) - np.ones(h.shape[1])
        return 0.5 * np.dot(alpha.T, np.dot(h, alpha)) - np.dot(alpha, np.ones(h.shape[1])), grad

    return dual_obj


# params = [ K*, C, p*, rbf]
def svm_linear(dtr, ltr, dte, params, pt):
    return score_mat(dtr, ltr, dte,  k=params[0], c=params[1], p=pt)

def svm_kernel_pol(dtr, ltr, dte, params, pt):
    return score_mat_kernel(dtr, ltr, dte, k=params[0], c=params[1], p=pt)

def svm_kernel_rbf(dtr, ltr, dte, params, pt):
    return score_mat_kernel(dtr, ltr, dte, k=params[0], c=params[1], p=pt, rbf=params[2])


# def plot_min_dcfs_svm(dtr, ltr, cfn, cfp):
#     c_values = np.logspace(-5, 5, num=51)
#     min_dcf_list = []
#     e = [0.5, 0.1, 0.9]
#     k = 0
#     for eff_p in e:
#         mins = []
#         for c in c_values:
#             mins.append(k_fold(dtr, ltr, 5, svm_linear, eff_p, cfn, cfp, seed=27, svm_params=[1, c, 0.5]))
#             k += 1
#             print(f"Iterazione {k}")
#         min_dcf_list.append(mins.copy())
#     for i in range(len(min_dcf_list)):
#         plt.plot(values, min_dcf_list[i], label=f"eff_p={e[i]}")
#     plt.xscale("log")
#     plt.xlabel("$c$")
#     plt.ylabel("minDCF")
#     plt.xlim([c_values[0], c_values[-1]])
#     plt.legend()
