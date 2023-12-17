import numpy as np
from general.utils import mrow

from models.logistic_regression.log_reg_training_models import LR
from general.utils import k_fold, predict_labels, get_confusion_matrix, compute_dcf, compute_min_dcf
from general.plotting import plot_bayes_error


def calibrate_scores(scores, l, prior, model_desc):
    return k_fold(scores, l, 5, LR(0, prior), p=prior, seed=27, model_desc=model_desc, calibration=True)


def fusion(scores_list, l, prior, model_desc):
    d = np.vstack(scores_list)
    return k_fold(d, l, 5, LR(0, prior), p=prior, seed=27, model_desc=model_desc, fusion=True)


def show_calibration_results(l, prior, cfn, cfp):
    best_model_scores = ["score_models/Tied/Tied_prior_None.npy",
                         "score_models/LR/LR_l_1e-05_pt_0.5.npy",
                         "score_models/Linear_SVM/raw/Linear_SVM_1_k_1_pt_0.9_.npy",
                         "score_models/Rbf_SVM/raw/Rbf_SVM_10.0_k_1_pt_0.5_gamma_0.001.npy"
                         "score_models/GMM/raw/pca/GMM_4__pca_11.npy",
                         "score_models/Tied_GMM_/z_score/Tied_GMM_8__pca_11_zscore.npy"]

    models_desc = ["TMVG", "LR", "Linear_SVM", "RBF_SVM", "GMM", "Tied_GMM"]

    calibrated_models_desc = ["Calibrated_TMVG",
                              "Calibrated_LR",
                              "Calibrated_Linear_SVM",
                              "Calibrated_RBF_SVM"
                              "Calibrated_GMM",
                              "Calibrated_Tied_GMM"]

    for i in range(len(best_model_scores)):
        print(calibrated_models_desc[i])
        uncalibrated_scores = np.load(best_model_scores[i])
        calibrated_scores = calibrate_scores(mrow(uncalibrated_scores), l, prior, calibrated_models_desc[i])
        plot_bayes_error(uncalibrated_scores, l, cfn, cfp, models_desc[i])
        plot_bayes_error(calibrated_scores, l, cfn, cfp, calibrated_models_desc[i])
        scores = [uncalibrated_scores, calibrated_scores]
        for j in range(len(scores)):
            dcfs = []
            for p in [0.1, 0.5, 0.9]:
                calibrated_pred = predict_labels(scores[j], -np.log(p / (1 - p)))
                conf_mat = get_confusion_matrix(calibrated_pred, l, 2)
                dcfs.append(compute_dcf(conf_mat, cfn, cfp, p))
            print(f"{"Uncalibrated" if i == 0 else "Calibrated"} => {dcfs}")


def show_fusion_results(l, prior, cfn, cfp):
    best_scores = ["score_models/Tied/Tied_prior_None.npy",
                   "score_models/LR/LR_l_1e-05_pt_0.5.npy",
                   "calibrated_score_models/Calibrated_Linear_SVM.npy",
                   "calibrated_score_models/Calibrated_RBF_SVM.npy",
                   "calibrated_score_models/Calibrated_GMM.npy",
                   "score_models/Tied_GMM_/z_score/Tied_GMM_8__pca_11_zscore.npy"]

    models_desc = ["TMVG", "LR", "Linear_SVM", "RBF_SVM", "GMM", "Tied_GMM"]

    fusions = ["Tied_GMM+RBF_SVM+LR",
               "GMM+RBF_SVM+LR",
               "Tied_GMM+Linear_SVM+LR",
               "GMM+Linear_SVM+LR",
               "Linear_SVM+LR",
               "RBF_SVM+LR",
               "Tied_GMM+Linear_SVM",
               "Tied_GMM+RBF_SVM",
               "GMM+Linear_SVM",
               "GMM+RBF_SVM"]

    for f in fusions:
        print(f)
        models = f.split("+")
        score_list = []
        for m in models:
            index = models_desc.index(m)
            score_list.append(np.load(best_scores[index]))
        fusion_scores = fusion(score_list, l, prior, f)
        # for p in [0.1, 0.5, 0.9]:
        #     print(f"{p} => {compute_min_dcf(fusion_scores, l, p, cfn, cfp)}, ", end="")
        plot_bayes_error(fusion_scores, l, cfn, cfp, f)
        print("\n")
