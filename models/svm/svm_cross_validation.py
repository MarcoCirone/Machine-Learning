from models.svm.svm_training_models import *
import numpy as np
from general.utils import k_fold, compute_min_dcf
from general.plotting import plot_min_dcfs_svm
import os

def cross_validation_for_svm(d, l, model, zscore):
    c_values = np.logspace(-5, 5, 31)
    pt = 0.5
    for c in c_values:
        print(f"k = {model.k} c = {c}, pt = {pt}")
        model.set_values(c, pt)
        _score = k_fold(d, l, 5, model, seed=27, zscore=zscore)


def compute_min_dcf_for_all(l, model, zscore, name):
    cfn = 1
    cfp = 1
    min_dcf_list = []
    prior = [0.5, 0.1, 0.9]
    c_values = np.logspace(-5, 5, 31)
    pt = 0.5
    for p in prior:
        min_dcf = []
        for c in c_values:
            print(f"prior = {p}, c = {c}, pt = {pt}")
            model.set_values(c, pt)
            score = np.load(f"score_models/{model.folder()}/{model.description()}{"_zscore" if zscore else ""}.npy")
            m_dcf = compute_min_dcf(score, l, p, cfn, cfp)
            min_dcf.append(m_dcf)
        min_dcf_list.append(min_dcf)
    if not os.path.exists("min_dcf_models/" + model.folder()):
        os.makedirs("min_dcf_models/" + model.folder())
    np.save(f"min_dcf_models/{model.folder()}/{name}_pt_{pt}{"_zscore" if zscore else ""}", min_dcf_list)


def plot_rbf_svm(name, zscore):
    _prior = [0.1, 0.5, 0.9]
    gamma = [0.001, 0.01, 0.1]
    k_values = [0.1, 1, 10]
    c_values = np.logspace(-5, 5, 31)
    pt = 0.5
    for g in gamma:
        if g == 0.001:
            g_name = "gamma_-3"
        else:
            if g == 0.01:
                g_name = "gamma_-2"
            else:
                g_name = "gamma_-1"
        min_dcf_list = []
        for k in k_values:
            rbf_svm = RbfSvm(g, k=k, preprocess="z_score" if zscore else "raw")
            min_dcf = np.load(f"min_dcf_models/{rbf_svm.folder()}/{name}_k_{k}_gamma_{g}__pt_{pt}{"_zscore" if zscore else ""}.npy")
            min_dcf_list.append(min_dcf[0])
        plot_min_dcfs_svm(min_dcf_list, f"rbf_svm_with_gamma_{g_name}_and_k", c_values)


def cv_linear_svm(d, l, zscore=False):
    k_values = [0.01, 0.1, 1, 10, 100]
    for k in k_values:
        l_svm = LinearSvm(k=k, preprocess="z_score" if zscore else"raw")
        cross_validation_for_svm(d, l, l_svm, zscore)
        compute_min_dcf_for_all(l, l_svm, zscore, f"Linear_SVM_k_{k}_")


def cv_pol_svm(d, l, zscore=False):
    k_values = [0.1, 1, 10]
    constant_values = [0, 1, 2]
    dim = 2
    for c in constant_values:
        print(f"constant = {c}")
        for k in k_values:
            pol_svm = PolSvm(c, dim, k=k, preprocess="z_score" if zscore else"raw")
            cross_validation_for_svm(d, l, pol_svm, zscore)
            compute_min_dcf_for_all(l, pol_svm, zscore, f"Polinomial_SVM_k_{k}_constant_{c}_dim_{dim}_")


def cv_rbf_svm(d, l, zscore=False):
    k_values = [0.1, 1, 10]
    gamma = [0.001, 0.01, 0.1]
    for g in gamma:
        print(f"gamma = {g}")
        for k in k_values:
            rbf_svm = RbfSvm(g, k=k, preprocess="z_score" if zscore else"raw")
            cross_validation_for_svm(d, l, rbf_svm, zscore)
            compute_min_dcf_for_all(l, rbf_svm, zscore, f"Rbf_SVM_k_{k}_gamma_{g}_")
        plot_rbf_svm(f"Rbf_SVM", zscore)


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
