from svm_training_models import *
import numpy as np
from general.utils import k_fold
from general.plotting import plot_min_dcfs_svm

def cross_validation_for_svm(d, l, model, zscore):
    cfn = 1
    cfp = 1

    c_values = np.logspace(-5, 5, 31)
    prior = [0.5, 0.1, 0.9]

    pt = 0.5
    min_dcf_list = []
    for p in prior:
        min_dcf = []
        for c in c_values:
            print(f"prior = {p}, c = {c}, pt = {pt}")
            model.set_values(c, pt)
            m_dcf = k_fold(d, l, 5, model, p, cfn, cfp, seed=27, zscore=zscore)
            min_dcf.append(m_dcf)
        min_dcf_list.append(min_dcf)
    plot_min_dcfs_svm(min_dcf_list, model.description, c_values)


def cv_linear_svm(d, l, zscore=False):
    k = 1
    l_svm = LinearSvm(k=k)
    cross_validation_for_svm(d, l, l_svm, zscore)


def cv_pol_svm(d, l, zscore=False):
    k = 1
    c = 1
    dim = 2
    pol_svm = PolSvm(c, dim, k=k)
    cross_validation_for_svm(d, l, pol_svm, zscore)

def cv_rbf_svm(d, l, zscore=False):
    k = 1
    gamma = [0.001, 0.01, 0.1]
    for g in gamma:
        print(f"gamma = {g}")
        rbf_svm = RbfSvm(g, k=k)
        cross_validation_for_svm(d, l, rbf_svm, zscore)

def cross_validation_for_all_svm(d, l):
    print("_____LINEAR SVM without Z Score_____")
    cv_linear_svm(d, l)
    print("_____LINEAR SVM with Z Score_____")
    cv_linear_svm(d, l, zscore=True)
    print("_____POLINOMIAL SVM without Z Score_____")
    cv_pol_svm(d, l)
    print("_____POLINOMIAL SVM with Z Score_____")
    cv_pol_svm(d, l, zscore=True)
    print("_____RBF SVM without Z Score_____")
    cv_rbf_svm(d, l)
    print("_____RBF SVM with Z Score_____")
    cv_rbf_svm(d, l, zscore=True)





