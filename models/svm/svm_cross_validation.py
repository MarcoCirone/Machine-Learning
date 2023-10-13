from svm_training_models import *
import numpy as np
from general.utils import k_fold


def cross_validation_for_svm(d, l, model):

    c_values = np.logspace(-5, 5, 31)
    prior = [0.5, 0.1, 0.9]
    prior_t = [None, 0.5, 0.1, 0.9]
    for p in prior:
        for pt in prior_t:
            for c in c_values:
                model.set_values(c, pt)
                min_dcf = k_fold(d, l, 5, model)


def cv_linear_svm(d, l):
    k = 1
    l_svm = LinearSvm(k=k)
    cross_validation_for_svm(d, l, l_svm)

def cv_pol_svm(d, l):
    k = 1
    c = 1
    d = 2
    pol_svm = PolSvm(c, d, k=k)
    cross_validation_for_svm(d, l, pol_svm)

def cv_rbf_svm(d, l):
    k = 1
    gamma = [0.001, 0.01, 0.1]
    for g in gamma:
        rbf_svm = RbfSvm(g, k=k)
        cross_validation_for_svm(d, l, rbf_svm)




