import numpy as np
from models.gaussian.gaussians_training_models import *
from models.gmm.gmm_training_models import *
from models.svm.svm_training_models import *
from models.logistic_regression.log_reg_training_models import *
from models.calibration.calibration_model import *
from general.plotting import plot_min_dcfs
from general.utils import k_fold, load, mcol, mrow

if __name__ == '__main__':
    dtr, ltr = load("train.txt")
    # dte, lte = load("Test.txt")
    
    prior = 0.5
    cfn = 1
    cfp = 1

    #min_dcf = k_fold(dtr, ltr, 5, LR, prior, cfn, cfp, seed=27)

    cross_val_log_reg(dtr, ltr, prior, cfn, cfp)

    # scores = mrow(numpy.load("score_models/LR/LR_l_0.0001_pt_0.5_prior_0.1.npy"))
    #
    # calibrate_scores(scores, ltr, prior, "LR_l_0.0001_pt_0.5_prior_0.1")
    # old_score_models = numpy.load("score_models/LR/LR_l_0.0001_pt_0.5_prior_0.1.npy")
    # calibrate_score = numpy.load("calibrated_score_models/LR_l_0.0001_pt_0.5_prior_0.1.npy")
    # plot_min_dcfs(dtr, ltr, cfn, cfp, svm_linear, pt=0.5, seed=27, svm_params=[1, 2])

    # min_dcf = k_fold(dtr, ltr, 5, mvg_loglikelihood_domain, prior, cfn, cfp, seed=27)
