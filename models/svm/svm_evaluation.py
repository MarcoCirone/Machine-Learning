from models.svm.svm_training_models import *
import numpy as np
from general.utils import compute_min_dcf
from general.plotting import plot_min_dcfs_svm_for_evaluation
from general.evaluation import get_evaluation_scores
import os

def evaluation_for_svm(d, l, model, zscore, dte, lte):
    c_values = np.logspace(-5, 5, 31)
    pt = 0.5
    for c in c_values:
        print(f"k = {model.k} c = {c}, pt = {pt}")
        model.set_values(c, pt)
        _score = get_evaluation_scores(d, l, dte, lte, model, zscore=zscore, model_desc=model.description())


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
            score = np.load(f"evaluation/scores/{model.description()}{"_zscore" if zscore else ""}.npy")
            m_dcf = compute_min_dcf(score, l, p, cfn, cfp)
            min_dcf.append(m_dcf)
        min_dcf_list.append(min_dcf)
    if not os.path.exists("min_dcf_models/" + model.folder()):
        os.makedirs("min_dcf_models/" + model.folder())
    np.save(f"evaluation/min_dicf/{name}_pt_{pt}{"_zscore" if zscore else ""}", min_dcf_list)


def plot_rbf_svm(name, zscore):

    gamma = [0.1, 0.01, 0.001]
    k = 1
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

        rbf_svm = RbfSvm(g, k=k, preprocess="z_score" if zscore else "raw")
        min_dcf_list1 = np.load(f"evaluation/min_dicf/{name}_k_{k}_gamma_{g_name}__pt_{pt}{"_zscore" if zscore else ""}.npy")
        min_dcf_list2 = np.load(f"min_dcf_models/{rbf_svm.folder()}/{name}_k_{k}_gamma_{g}__pt_{pt}{"_zscore" if zscore else ""}.npy")
        plot_min_dcfs_svm_for_evaluation(min_dcf_list2, min_dcf_list1, f"evaluation_{name}", c_values)

def eval_linear_svm(d, l, dte, lte, zscore=False):
    k = 1
    l_svm = LinearSvm(k=k, preprocess="z_score" if zscore else"raw")
    evaluation_for_svm(d, l, l_svm, zscore, dte, lte)
    compute_min_dcf_for_all(l, l_svm, zscore, f"Linear_new_SVM_k_{k}_")
    plot_min_dcfs_svm_for_evaluation(score, score_ev, f"Linear_SVM", values=np.logspace(-5, 5, 31))

def eval_rbf_svm(d, l, dte, lte, zscore=False):
    k = 1
    gamma = [0.1, 0.01, 0.001]
    for g in gamma:
        print(f"gamma = {g}")
        rbf_svm = RbfSvm(g, k=k, preprocess="z_score" if zscore else"raw")
        evaluation_for_svm(d, l, rbf_svm, zscore, dte, lte)
        compute_min_dcf_for_all(lte, rbf_svm, zscore, f"Rbf_SVM_k_{k}_gamma_{g}_")
    plot_rbf_svm(f"Rbf_SVM", zscore)


def evaluation_for_all_svm(d, l, dte=None, lte=None):
    print("_____LINEAR SVM without Z Score_____")
    eval_linear_svm(d, l, dte, lte)
    print("_____RBF SVM without Z Score_____")
    eval_rbf_svm(d, l, dte, lte)
