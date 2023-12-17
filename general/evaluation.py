from models.gaussian.gaussians_training_models import MVGTied
from models.logistic_regression.log_reg_training_models import LR
from models.svm.svm_training_models import LinearSvm, RbfSvm
from models.gmm.gmm_training_models import GMM, GMMTied
from models.calibration.calibration_model import calibrate_scores
from general.utils import z_score, pca, mrow, compute_min_dcf
import numpy as np
import os


def get_evaluation_scores(dtr, ltr, dte, lte, model, p=None, pca_m=None, zscore=False, calibration=False, fusion=False, model_desc=None):
    if zscore:
        dtr, dte = z_score(dtr, dte)

    if pca_m is not None:
        # PCA
        p1 = pca(dtr, pca_m)
        dtr = np.dot(p1.T, dtr)
        dte = np.dot(p1.T, dte)

    model.set_data(dtr, ltr, dte, lte)
    model.train()
    scores = model.get_scores()
    if not os.path.exists("evaluation"):
        os.makedirs("evaluation")
    if not calibration:
        if not fusion:
            if not os.path.exists("evaluation/scores"):
                os.makedirs("evaluation/scores")
            np.save(f"evaluation/scores/{model_desc}", scores)
        else:
            if not os.path.exists("evaluation/fusion_scores"):
                os.makedirs("evaluation/fusion_scores")
            np.save(f"evaluation/fusion_scores/{model_desc}", scores)
    else:
        scores -= np.log(p / (1 - p))
        if not os.path.exists("evaluation/calibrated_scores"):
            os.makedirs("evaluation/calibrated_scores")
        np.save(f"evaluation/calibrated_scores/{model_desc}", scores)
    return scores


def evaluate_best_models(dtr, ltr, dte, lte, prior, cfn, cfp):
    # for each model we store the model itself, the dimension of pca, the presence of z-score, if calibration is used
    # and their description
    best_models = [#(MVGTied(), None, False, False, "TMVG"),
                   #(LR(10**(-5), 0.5), None, False, False, "LR"),
                   (LinearSvm(1, 1, 0.9), None, False, True, "Linear_SVM"),
                   (RbfSvm(10**(-3), 10, 1, 0.5), None, False, True, "RBF_SVM"),
                   (GMM(4), 11, False, True, "GMM"),
                   ]#(GMMTied(8), 11, True, False, "Tied_GMM")]

    for model in best_models:
        print(model[4])
        # scores = get_evaluation_scores(dtr, ltr, dte, lte, model[0], p=prior, pca_m=model[1], zscore=model[2],
        #                               calibration=False, fusion=False, model_desc=model[4])
        scores = np.load(f"evaluation/scores/{model[4]}.npy")
        if model[3]:
            #scores = get_evaluation_scores(mrow(np.load(f"calibrated_score_models/Calibrated_{model[4]}.npy")), ltr,
            #                               mrow(scores), lte, LR(0, prior), p=prior, pca_m=None, zscore=None,
            #                               calibration=True, fusion=False, model_desc=model[4])
            scores = np.load(f"evaluation/calibrated_scores/{model[4]}.npy")
        for p in [0.1, 0.5, 0.9]:
            print(f"{p} => {compute_min_dcf(scores, lte, p, cfn, cfp)} ", end="")
        print("\n")

def evaluate_fusion(ltr, lte, prior, cfn, cfp):
    fusion_models = []