import numpy as np
import math
import scipy.special as sp
import sys
sys.path.append("../../")
from utils import logpdf_GAU_ND, mrow, mcol
from training.gaussian.gaussians_models import score_as_vec

def logpdf_GMM(x, gmm):
    s = np.zeros([len(gmm), x.shape[1]])
    for i in range(len(gmm)):
        s[i, :] = logpdf_GAU_ND(x, gmm[i][1], gmm[i][2])+np.log(gmm[i][0])      # gmm[i]=[w, mu, cov]
    logdens = sp.logsumexp(s, axis=0)
    return s, mrow(logdens)

def comp_stats(res, x):
    zg = np.zeros(res.shape[0])
    fg = np.zeros([res.shape[0], x.shape[0]])
    sg = []
    for g in range(res.shape[0]):
        sg.append(0)
        for i in range(x.shape[1]):
            zg[g] += res[g, i]
            fg[g] += res[g, i]*x[:, i]
            xt = mcol(x[:, i])
            sg[g] += res[g, i]*np.dot(xt, xt.T)
    return zg, fg, sg

def comp_params(res, x, psi, diag):
    zg, fg, sg = comp_stats(res, x)
    new_gmm = []
    for g in range(res.shape[0]):
        w = zg[g]/zg.sum()
        mu = mcol(fg[g]/zg[g])
        cov = sg[g]/zg[g]-np.dot(mu, mu.T)
        cov = comp_norm_diag(cov, diag)
        u, s, _ = np.linalg.svd(cov)
        s[s < psi] = psi
        cov = np.dot(u, mcol(s)*u.T)
        new_gmm.append((w, mu, cov))
    return new_gmm

def comp_params_tied(res, x, psi, diag):
    zg, fg, sg = comp_stats(res, x)
    w = np.zeros(res.shape[0])
    mu = np.zeros([res.shape[0], x.shape[0]])
    cov = np.zeros([x.shape[0], x.shape[0]])
    new_gmm = []
    for g in range(res.shape[0]):
        w[g] = zg[g] / zg.sum()
        mu[g] = fg[g] / zg[g]
        mt = mcol(mu[g])
        cov_new = sg[g] / zg[g] - np.dot(mt, mt.T)
        cov_new = comp_norm_diag(cov_new, diag)
        cov += zg[g] * cov_new
    cov /= x.shape[1]
    u, s, _ = np.linalg.svd(cov)
    s[s < psi] = psi
    cov = np.dot(u, mcol(s) * u.T)

    for i in range(res.shape[0]):
        new_gmm.append((w[i], mcol(mu[i]), cov))
    return new_gmm

def comp_norm_diag(cov_new, diag):
    if diag:
        return cov_new*np.eye(cov_new.shape[0])
    return cov_new

def EM_algorithm(x, gmm, stop, psi, tied, diag):
    log_vec = []
    gmm_vec = [gmm]
    s, logdens = logpdf_GMM(x, gmm)
    log_vec.append(logdens.sum() / x.shape[1])
    while True:
        # E step
        res = np.exp(s - logdens)
        # M step
        if tied:
            new_gmm = comp_params_tied(res, x, psi, diag)
        else:
            new_gmm = comp_params(res, x, psi, diag)
        gmm_vec.append(new_gmm)
        new_s, new_logdens = logpdf_GMM(x, new_gmm)
        log_vec.append(new_logdens.sum() / x.shape[1])
        if log_vec[-1] - log_vec[-2] < stop:
            break
        else:
            logdens = new_logdens
            s = new_s
            continue
    return log_vec, gmm_vec


def LBG_algorithm(alpha, g_num, x, stop, psi, diag, tied):
    mu = mcol(x.mean(axis=1))
    cov = 1 / (x.shape[1]) * np.dot(x - mu, (x - mu).T)

    u, s, _ = np.linalg.svd(comp_norm_diag(cov, diag))
    s[s < psi] = psi
    cov = np.dot(u, mcol(s) * u.T)

    gmm = [(1.0, mu, cov)]

    while len(gmm) < g_num:
        gmm_1 = []
        for i in range(len(gmm)):
            u, s, vh = np.linalg.svd(gmm[i][2])
            d = u[:, 0:1] * s[0] ** 0.5 * alpha
            w = gmm[i][0] / 2
            mu1 = gmm[i][1] + d
            mu2 = mu1 - 2 * d
            gmm_1.append((w, mu1, gmm[i][2]))
            gmm_1.append((w, mu2, gmm[i][2]))
        _, gmm_vec = EM_algorithm(x, gmm_1, stop, psi, tied, diag)
        gmm = gmm_vec[-1]
    return gmm

def score_mat(dtr, ltr, dte, g_num, diag=False, tied=False):
    score = np.zeros(dte.shape[1])
    gmm = gmm_model(dtr, ltr, g_num, diag, tied)
    for j in range(dte.shape[1]):
        tmp_score = []
        for i in range(len(gmm)):
            _, logdens = logpdf_GMM(mcol(dte[:, j]), gmm[i])
            tmp_score.append(logdens)
        score[j] = tmp_score[1]-tmp_score[0]
    return score

def gmm_model(dtr, ltr, g_num, diag, tied, alpha=0.1, psi=0.01, stop=10**(-6)):
    gmm = []
    for i in range(2):
        print("gmm "+str(i))
        x = dtr[:, ltr == i]
        gmm.append(LBG_algorithm(alpha, g_num, x, stop, psi, diag, tied))
    return gmm

def gmm_loglikelihood_domain(dtr, ltr, dte, g_num):
    return score_mat(dtr, ltr, dte, g_num)

def gmm_loglikelihood_naiveBayes(dtr, ltr, dte, g_num):
    return score_mat(dtr, ltr, dte, g_num, diag=True)

def gmm_loglikelihood_TiedCovariance(dtr, ltr, dte, g_num):
    return score_mat(dtr, ltr, dte, g_num, tied=True)

def gmm_loglikelihood_TiedNaiveByes(dtr, ltr, dte, g_num):
    return score_mat(dtr, ltr, dte, g_num, tied=True, diag=True)



