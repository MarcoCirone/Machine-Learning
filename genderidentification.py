#  import numpy as np
import scipy
from models.gaussian.gaussians_training_models import *
from models.gmm.gmm_training_models import *
from general.utils import load, k_fold

if __name__ == '__main__':
    dtr, ltr = load("Train.txt")
    dte, lte = load("Test.txt")
    
    prior = 0.5
    cfn = 1
    cfp = 1

    min_dcf = k_fold(dtr, ltr, 5, gmm_loglikelihood_TiedDiag, prior, cfn, cfp, seed=27, g_num=4)
    # compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/MVG_Results.txt", MVG_log_likelihood_domain)
    # compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/TVG_Results.txt", TVG_log_likelihood_domain)
    # compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    # compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    # kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/MVG_Results.txt", MVG_log_likelihood_domain)
    # kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/TVG_Results.txt", TVG_log_likelihood_domain)
    # kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    # kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    # LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/MVG_Results.txt", MVG_log_likelihood_domain)  FATTO
    # LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/TVG_Results.txt", TVG_log_likelihood_domain)  FATTO
    # LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    # LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    # hist(DTR, LTR)
    # scatter3D(DTR, LTR)
    # D, L = load("iris.csv")
    
    # P1 = PCA(DTR, 6)
    # y = np.dot(P1.T, DTR)
    # scatter3D(y, LTR)
    # hist(y, LTR)
    
    # P2 = LDA(y, LTR, 3)
    # y = np.dot(P2.T, y)
    
    # mu0 = mcol(y[:, LTR==0].mean(axis=1))
    # C0 = np.dot(y[:, LTR==0]-mu0, (y[:, LTR==0]-mu0).T)/y[:, LTR==0].shape[1]
        
    # mu1 = mcol(y[:, LTR==1].mean(axis=1))
    # C1 = np.dot(y[:, LTR==1]-mu1, (y[:, LTR==1]-mu1).T)/y[:, LTR==1].shape[1]
    
    # print(np.abs(C0-C1).max())
    
    # hist(y, LTR)
    # scatter_3d(y, LTR)
