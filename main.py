import numpy as np
from general.plotting import *
from general.utils import load
from general.evaluation import evaluate_best_models, evaluate_fusion, evaluate_calibration
from models.logistic_regression.log_reg_evaluation import evaluate_LR
from models.logistic_regression.log_reg_training_models import cross_val_log_reg, cross_val_quad_log_reg
from models.svm.svm_cross_validation import cross_validation_for_all_svm
from models.svm.svm_evaluation import evaluation_for_all_svm
from models.gmm.gmm_cross_validation import cross_validation_for_all_gmm
from models.gmm.gmm_evaluation import evaluation_for_all_gmm
from models.gaussian.gaussian_all_model import gaussians
from models.calibration.calibration_model import show_calibration_results, show_fusion_results
import scipy.special as sp

if __name__ == '__main__':
    dtr, ltr = load("train.txt")
    dte, lte = load("Test.txt")
    
    prior = 0.5
    cfn = 1
    cfp = 1

    # dataset analysis
    labels = ["Male", "Female", "All"]
    hist(dtr, ltr, labels, name="histograms_")
    hist(lda(dtr, ltr), ltr, labels, name="histograms_lda_")
    scatter_2d(dtr, ltr, labels)
    plot_pca(dtr)
    plot_heatmaps(dtr, ltr, labels)

    # VALIDATION PHASE
    gaussians(dtr, ltr)     # gaussian models
    cross_val_log_reg(dtr, ltr, prior, cfn, cfp)    # logistic regression models
    cross_val_quad_log_reg(dtr, ltr, prior, cfn, cfp)   # quadratic logistic regression models
    cross_validation_for_all_svm(dtr, ltr)      # SVM models
    cross_validation_for_all_gmm(dtr, ltr)      # GMM models

    show_calibration_results(ltr, prior, cfn, cfp)      # Calibration
    show_fusion_results(ltr, prior, cfn, cfp)       # Fusion

    scores = [
        np.load("score_models/Tied/Tied_prior_None.npy"),
        np.load("score_models/LR/LR_l_1e-05_pt_0.5.npy"),
        np.load("calibrated_score_models/Calibrated_RBF_SVM.npy"),
        np.load("score_models/Tied_GMM_/z_score/Tied_GMM_8__pca_11_zscore.npy")
    ]
    model_labels = ["TMVG", "LR", "RBF SVM", "Tied GMM"]
    plot_det_roc(scores, ltr, model_labels, train=True)

    # EVALUATION PHASE
    evaluate_best_models(dtr, ltr, dte, lte, prior, cfn, cfp)
    evaluate_fusion(ltr, lte, cfn, cfp)

    new_scores = [
        np.load("evaluation/scores/TMVG.npy"),
        np.load("evaluation/scores/LR.npy"),
        np.load("evaluation/scores/RBF_SVM.npy"),
        np.load("evaluation/scores/GMM.npy")
    ]
    new_model_labels = ["TMVG", "LR", "RBF SVM", "GMM"]
    plot_det_roc(new_scores, lte, model_labels, train=False)

    evaluate_LR(dtr, ltr, dte, lte, cfn, cfp)
    evaluation_for_all_svm(dtr, ltr, dte, lte)
    evaluation_for_all_gmm(dtr, ltr, dte, lte)
