import numpy as np
from models.gaussian.gaussians_training_models import *
from models.gmm.gmm_training_models import *
from models.svm.svm_training_models import *
from general.utils import mcol
from general.utils import k_fold, load

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

def comp_err_rate(PL, LTE, DTE):
    mask = PL == LTE

    acc = mask.sum(0) / DTE.shape[1]
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

# def load(path):
#     with open(path) as f:
#         list = []
#         labelList = []
#
#         for r in f:
#             try:
#                 vett = r.split(",")[0:4]
#                 labelList.append(hLabels[r.split(",")[4][:-1]])
#                 list.append(mcol(np.array([float(i) for i in vett])))
#
#             except:
#                 pass
#     return np.hstack(list), np.array(labelList, dtype=np.int32)


if __name__ == '__main__':
    dtr, ltr = load("train.txt")
    # dte, lte = load("Test.txt")

    # D = D[:, L != 0] # We remove setosa from D
    # L = L[L!=0] # We remove setosa from L
    # L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    # (dtr, ltr), (dte, lte) = split_db_2to1(D, L)
    
    prior = 1/11
    cfn = 1
    cfp = 1
    # SVM LINEARI
    # prior 0.5
    # 0.5 131, 0.9 0.127, 0.1 0.127
    # prior 0.1
    # 0.5 0.318, 0.9 0.368, 0.1 0.31
    # prior 0.9
    # 0.5 0.389,0.9 0.341,0.1 0.0364
    # SVM POLINOMIALI
    # prior 0.5
    # 0.5 0.93, 0.9 0.98, 0.1 0.85
    min_dcf = k_fold(dtr, ltr, 5, svm_linear, prior, cfn, cfp, seed=0, svm_params=[1, 10, None])
    # score = svm_kernel_rbf(dtr, ltr, dte, [1, 1, None, 1])
    # predicted = np.where(score > 0, 1, 0)
    # err = comp_err_rate(predicted, lte, dte)
    # print(err)




    # min_dcf = k_fold(dtr, ltr, 5, mvg_loglikelihood_domain, prior, cfn, cfp, seed=27)

    # compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/MVG_Results.txt", MVG_log_likelihood_domain)
    # compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/TVG_Results.txt", TVG_log_likelihood_domain)
    # compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    # compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    # kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/MVG_Results.txt", MVG_log_likelihood_domain)
    # kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/TVG_Results.txt", TVG_log_likelihood_domain)
    # kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    # kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
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
