import numpy as np
from models.gaussian.gaussians_training_models import *
from models.gmm.gmm_training_models import *
from models.svm.svm_training_models import *
from models.svm.svm_cross_validation import *
from models.logistic_regression.log_reg_training_models import *
from models.gaussian.gaussian_all_model import *
from general.plotting import *
from general.utils import k_fold, load, mcol
from models.calibration.calibration_model import *
from general.utils import k_fold, mcol, mrow
import scipy.special as sp

def logpdf_GMM1(x, gmm):
    s=numpy.zeros([len(gmm), x.shape[1]])
    for i in range(len(gmm)):
        s[i,:]=logpdf_GAU_ND(x,gmm[i][1], gmm[i][2])+ numpy.log(gmm[i][0])
    logdens = sp.logsumexp(s, axis=0)
    return s, mrow(logdens)

def predict_label(DTE,LTE, gmm, prior):
    PL=np.zeros(LTE.shape)
    for j in range(DTE.shape[1]):
        class_posterior=np.empty(len(prior))
        for i in range(len(gmm)):
            _, logdens= logpdf_GMM1(mcol(DTE[:,j]), gmm[i])
            class_posterior[i]= np.log(prior[i])+logdens
        PL[j]=class_posterior.argmax()
    return PL

def comp_err_rate(PL, LTR, DTR):
    mask = PL == LTR
    acc = mask.sum(0) / DTR.shape[1]
    return (1 - acc) * 100

attr = {
    0: "sepal length",
    1: "sepal width",
    2: "petal length",
    3: "petal width"
}

hLabels = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def load(path):
    with open(path) as f:
        list = []
        labelList = []

        for r in f:
            try:
                vett = r.split(",")[0:4]
                labelList.append(hLabels[r.split(",")[4][:-1]])
                list.append(mcol(np.array([float(i) for i in vett])))

            except:
                pass
    return np.hstack(list), np.array(labelList, dtype=np.int32)


if __name__ == '__main__':
    # dtr, ltr = load("train.txt")
    # dte, lte = load("Test.txt")
    
    prior=[0.5]*3
    cfn = 1
    cfp = 1
    labels = ["Male", "Female", "All"]
    # cross_validation_for_all_svm(dtr, ltr)
    D, L = load("iris.csv")
    (dtr, ltr), (DTE, LTE) = split_db_2to1(D, L)
    gmm = GMMTied(g_num=8)
    gmm.set_data(dtr, ltr,None,None)
    gmm.train()
    PL =predict_label(DTE, LTE, gmm.gmm, prior)
    err = comp_err_rate(PL, LTE, DTE)
    # score = k_fold(dtr, ltr, 5, gmm, seed=27)
    # min_dcf_models = compute_min_dcf(score, ltr, prior, cfn, cfp)


    #min_dcf = k_fold(dtr, ltr, 5, LR, prior, cfn, cfp, seed=27)
    #gaussians(dtr, ltr)

    # cross_val_log_reg(dtr, ltr, prior, cfn, cfp)
    # cross_val_quad_log_reg(dtr, ltr, prior, cfn, cfp)

    # scores = k_fold(dtr, ltr, 5, LR(0, 0.9), seed=27, zscore=True)
    # np.save("log_reg_scores", scores)
    # scores = np.load("log_reg_scores.npy")
    # scores1 = calibrate_scores(mrow(scores), ltr, 0.5, "LR")
    # plot_bayes_error(scores, ltr, cfn, cfp, "LR_uncalibrated")
    # plot_bayes_error(scores1, ltr, cfn, cfp, "LR_calibrated")

    # scores = k_fold(dtr, ltr, 5, LinearSvm(c=10, k=1, pt=0.5), seed=27, zscore=True)
    # np.save("svm_scores", scores)
    # scores = np.load("svm_scores.npy")
    # scores1 = calibrate_scores(mrow(scores), ltr, 0.5, "SVM")
    # plot_bayes_error(scores, ltr, cfn, cfp, "SVM_uncalibrated")
    # plot_bayes_error(scores1, ltr, cfn, cfp, "SVM_calibrated")

    # scores = k_fold(dtr, ltr, 5, MVGTied(), seed=27, pca_m=12)
    # np.save("TMVG_scores", scores)
    # scores = np.load("TMVG_scores.npy")
    # scores1 = calibrate_scores(mrow(scores), ltr, 0.5, "TMVG")
    # plot_bayes_error(scores, ltr, cfn, cfp, "TMVG_uncalibrated")
    # plot_bayes_error(scores1, ltr, cfn, cfp, "TMVG_calibrated")

    # svm_scores = np.load("calibrated_score_models/SVM.npy")
    # lr_scores = np.load("calibrated_score_models/LR.npy")
    #
    # new_scores = fusion([svm_scores, lr_scores], ltr, 0.5, "SVM+LR")
    # np.save("svm+lr", new_scores)
    # plot_bayes_error(new_scores, ltr, cfn, cfp, "SVM+LR_uncalibrated")
    #
    # # scores = mrow(np.load("score_models/LR/LR_l_0.0001_pt_0.5_prior_0.1.npy"))
    # calibrate_scores(scores, ltr, prior, "LR_l_0.0001_pt_0.5_prior_0.1")
    # old_score_models = np.load("score_models/LR/LR_l_0.0001_pt_0.5_prior_0.1.npy")
    # calibrate_score = np.load("calibrated_score_models/LR_l_0.0001_pt_0.5_prior_0.1.npy")
    # plot_min_dcfs(dtr, ltr, cfn, cfp, svm_linear, pt=0.5, seed=27, svm_params=[1, 2])
    # plot_heatmaps(dtr, ltr, labels)
    # plot_pca(dtr)
    # scatter_2d(dtr, ltr, labels)
    # hist(dtr, ltr,labels)

    # min_dcf = k_fold(dtr, ltr, 5, LR, prior, cfn, cfp, seed=27, pt=0.5, reg_term=0)
    # plot_min_dcfs(dtr, ltr, cfn, cfp, LinearSvm, pt=0.5, seed=27, svm_params=[1, 2])

    # min_dcf = k_fold(dtr, ltr, 5, mvg_loglikelihood_domain, prior, cfn, cfp, seed=27)
