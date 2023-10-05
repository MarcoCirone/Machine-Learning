# import numpy as np
import scipy
from training.gaussian.gaussians_models import *
from utils import load


def PCA(D, n):
    mu = mcol(D.mean(axis = 1))
    DC = D - mu
    C = np.dot(DC, DC.T)/D.shape[1]
    _, U = np.linalg.eigh(C)
    return U[:, ::-1][:, 0:n]

def LDA(D, L, n):
    mu = mcol(D.mean(axis = 1))
    SB = np.zeros((D.shape[0], D.shape[0]))
    SW = np.zeros((D.shape[0], D.shape[0]))
    for c in range(L.max()+1):
        D_c = D[:, L==c]
        mu_c = mcol(D_c.mean(axis = 1))
        SB += D_c.shape[1] * np.dot((mu_c - mu), (mu_c - mu).T)
        SW += np.dot((D_c - mu_c), (D_c - mu_c).T)
    SB /= D.shape[1]
    SW /= D.shape[1]
    _,U=scipy.linalg.eigh(SB, SW)
    W=U[:, ::-1][:, 0:n]
    return W

# def compute_results(DTR, LTR, DTE, LTE, Prior, file_name, model):
#     file = open(file_name, "w")
#
#     content = ""
#
#     for i in range(DTR.shape[0], 0, -1):
#         if i != DTR.shape[0]:
#             P1 = PCA(DTR, i)
#             nDTR = np.dot(P1.T, DTR)
#             nDTE = np.dot(P1.T, DTE)
#         else:
#             nDTR = DTR
#             nDTE = DTE
#         for j in range(i, 0, -1):
#             print(f"i={i} j={j}\n")
#             if j != i:
#                 P2 = LDA(nDTR, LTR, j)
#                 nDTR = np.dot(P2.T, nDTR)
#                 nDTE = np.dot(P2.T, nDTE)
#             logS = model(nDTR, LTR, nDTE)
#             PL = compute_predicted_labels_log(logS, Prior)
#
#             CL = PL == LTE
#             acc = CL.sum(0)/CL.shape[0]
#             err_rate = (1-acc)*100
#             content += f"Dim_after_PCA:{i},Dim_after_LDA:{j},Error_Rate:{err_rate}%\n"
#
#     file.write(content)
#     file.close()

if __name__ == '__main__':
    dtr, ltr = load("Train.txt")
    dte, lte = load("Test.txt")
    
    Prior = 0.5
    s_matrix= mvg_loglikelihood_domain(dtr, ltr, dte)
    #compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/MVG_Results.txt", MVG_log_likelihood_domain)
    #compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/TVG_Results.txt", TVG_log_likelihood_domain)
    #compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    #compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    #kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/MVG_Results.txt", MVG_log_likelihood_domain)
    #kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/TVG_Results.txt", TVG_log_likelihood_domain)
    #kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    #kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    #LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/MVG_Results.txt", MVG_log_likelihood_domain)  FATTO
    #LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/TVG_Results.txt", TVG_log_likelihood_domain)  FATTO
    #LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    #LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    #hist(DTR, LTR)
    #scatter3D(DTR, LTR)
    #D, L = load("iris.csv")
    
    #P1 = PCA(DTR, 6)
    #y = np.dot(P1.T, DTR)
    #scatter3D(y, LTR)
    #hist(y, LTR)
    
    #P2 = LDA(y, LTR, 3)
    #y = np.dot(P2.T, y)
    
    #mu0 = mcol(y[:, LTR==0].mean(axis=1))
    #C0 = np.dot(y[:, LTR==0]-mu0, (y[:, LTR==0]-mu0).T)/y[:, LTR==0].shape[1]
        
    #mu1 = mcol(y[:, LTR==1].mean(axis=1))
    #C1 = np.dot(y[:, LTR==1]-mu1, (y[:, LTR==1]-mu1).T)/y[:, LTR==1].shape[1]
    
    #print(np.abs(C0-C1).max())
    
    #hist(y, LTR)
    #scatter3D(y, LTR)