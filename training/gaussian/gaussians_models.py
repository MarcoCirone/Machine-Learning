import numpy as np
import sys
sys.path.append("../../")
from utils import mcol, logpdf_GAU_ND

def mu_and_covariance(d, l, n):
    mu=[]
    cov=[]
    for i in range(n):
        dc=d[:, l==i] #samples for class i
        mu.append(dc.mean(1)) #mean for class i
        center_data= dc- mcol(dc.mean(1))     #center dataset
        cov.append(np.dot(1/(dc.shape[1])*center_data,center_data.T)) #compute covariance matrix
    return mu, cov

def score_mat(dtr, ltr, dte, tied=False, diag=False):
    n_class= np.max(ltr)+1
    mu, cov= mu_and_covariance(dtr, ltr, n_class)
    score=[]
    if tied:

        tied_cov=np.zeros([dtr.shape[0], dtr.shape[0]])
        for i in range(n_class):
            dc = dtr[:, ltr == i]  # samples for class i
            tied_cov+=np.dot((dc-mcol(mu[i])), (dc-mcol(mu[i])).T)
        tied_cov/= dtr.shape[1]
        for i in range(n_class):
            if diag:
                score.append(logpdf_GAU_ND(dte, mcol(mu[i]), tied_cov * np.eye(tied_cov.shape[0])))
            else:
                score.append(logpdf_GAU_ND(dte, mcol(mu[i]), tied_cov))
    else:
        for i in range(n_class):
            if diag:
                score.append(logpdf_GAU_ND(dte, mcol(mu[i]), [i] * np.eye(cov[i].shape[0])))
            else:
                score.append(logpdf_GAU_ND(dte, mcol(mu[i]), cov[i]))

    return score_as_vec(np.vstack(score))

def score_as_vec(score_matrix):
    com = np.zeros(score_matrix.shape[1])
    for i in range(score_matrix.shape[1]):
        com[i] = score_matrix[1][i] - score_matrix[0][i]
    return com



def mvg_loglikelihood_domain(dtr, ltr, dte):
    return score_mat(dtr, ltr, dte)
def mvg_loglikelihood_naiveBayes(dtr, ltr, dte):
    return score_mat(dtr, ltr, dte, diag=True)
def mvg_loglikelihood_TiedCovariance(dtr, ltr, dte):
    return score_mat(dtr, ltr, dte,  tied=True)
def mvg_loglikelihood_TiedNaiveByes(dtr, ltr, dte):
    return score_mat(dtr, ltr, dte,  tied=True, diag=True)