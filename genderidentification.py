import numpy as np
from general.plotting import *
from general.utils import load
from general.evaluation import evaluate_best_models
from models.calibration.calibration_model import show_calibration_results, show_fusion_results
import scipy.special as sp

if __name__ == '__main__':
    dtr, ltr = load("train.txt")
    dte, lte = load("Test.txt")
    
    prior = 0.5
    cfn = 1
    cfp = 1
    labels = ["Male", "Female", "All"]

    evaluate_best_models(dtr, ltr, dte, lte, prior, cfn, cfp)

    show_fusion_results(ltr, prior, cfn, cfp)

    # cross_validation_for_all_svm(dtr, ltr)

    # gmm = GMMDiag(g_num=4)
    # cross_validation_for_all_gmm(dtr, ltr)

    # for z_score in [True]:
    #     for pt in [0.1, 0.9]:
    #         print(f"pt={pt}, zscore={z_score}")
    #         k_fold(dtr, ltr, 5, QLR(10**-5, pt), seed=27, zscore=z_score)

    # plot_heatmaps(dtr, ltr, labels)
    # plot_pca(dtr)
    # scatter_2d(dtr, ltr, labels)
    # hist(dtr, ltr,labels)

    # min_dcf = k_fold(dtr, ltr, 5, LR, prior, cfn, cfp, seed=27, pt=0.5, reg_term=0)
    # plot_min_dcfs(dtr, ltr, cfn, cfp, LinearSvm, pt=0.5, seed=27, svm_params=[1, 2])

    # min_dcf = k_fold(dtr, ltr, 5, mvg_loglikelihood_domain, prior, cfn, cfp, seed=27)
