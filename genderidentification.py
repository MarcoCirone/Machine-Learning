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

if __name__ == '__main__':
    dtr, ltr = load("train.txt")
    # dte, lte = load("Test.txt")
    
    prior = 0.5
    cfn = 1
    cfp = 1
    labels = ["Male", "Female", "All"]
    # cross_validation_for_all_svm(dtr, ltr)

    gmm = GMMDiag(g_num=4)


    score = k_fold(dtr, ltr, 5, gmm, seed=27)
    min_dcf_models = compute_min_dcf(score, ltr, prior, cfn, cfp)


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
