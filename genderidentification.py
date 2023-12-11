import numpy as np
from models.gaussian.gaussians_training_models import *
from models.gmm.gmm_training_models import *
from models.gmm.gmm_cross_validation import *
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

    # gmm = GMMDiag(g_num=4)
    # cross_validation_for_all_gmm(dtr, ltr)

    # for z_score in [True]:
    #     for pt in [0.1, 0.9]:
    #         print(f"pt={pt}, zscore={z_score}")
    #         k_fold(dtr, ltr, 5, QLR(10**-5, pt), seed=27, zscore=z_score)

    best_model_scores = ["score_models/Tied/Tied_prior_None.npy",
                         "score_models/LR/LR_l_1e-05_pt_0.5.npy",
                         "score_models/Linear_SVM/raw/Linear_SVM_1_k_1_pt_0.9_.npy",
                         "score_models/Rbf_SVM/raw/Rbf_SVM_10.0_k_1_pt_0.5_gamma_0.001.npy",
                         "score_models/GMM/raw/pca/GMM_4__pca_11.npy",
                         "score_models/Tied_GMM_/z_score/Tied_GMM_8__pca_11_zscore.npy"]

    models_desc = ["TMVG", "LR", "Linear_SVM", "RBF_SVM", "GMM", "Tied_GMM"]

    calibrated_models_desc = ["Calibrated_TMVG",
                              "Calibrated_LR",
                              "Calibrated_Linear_SVM",
                              "Calibrated_RBF_SVM",
                              "Calibrated_GMM",
                              "Calibrated_Tied_GMM"]

    for i in range(len(best_model_scores)):
        print(calibrated_models_desc[i])
        uncalibrated_scores = np.load(best_model_scores[i])
        calibrated_scores = calibrate_scores(mrow(uncalibrated_scores), ltr, prior, calibrated_models_desc[i])
        plot_bayes_error(uncalibrated_scores, ltr, cfn, cfp, models_desc[i])
        plot_bayes_error(calibrated_scores, ltr, cfn, cfp, calibrated_models_desc[i])

    # plot_heatmaps(dtr, ltr, labels)
    # plot_pca(dtr)
    # scatter_2d(dtr, ltr, labels)
    # hist(dtr, ltr,labels)

    # min_dcf = k_fold(dtr, ltr, 5, LR, prior, cfn, cfp, seed=27, pt=0.5, reg_term=0)
    # plot_min_dcfs(dtr, ltr, cfn, cfp, LinearSvm, pt=0.5, seed=27, svm_params=[1, 2])

    # min_dcf = k_fold(dtr, ltr, 5, mvg_loglikelihood_domain, prior, cfn, cfp, seed=27)
