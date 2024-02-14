import numpy as np
import math
import scipy
import os


def pca(d, n, eigen_values=False):
    mu = mcol(d.mean(axis=1))
    dc = d - mu
    cov = np.dot(dc, dc.T)
    cov = cov / float(dc.shape[1])
    s, u = np.linalg.eigh(cov)
    if eigen_values:
        return s
    return u[:, ::-1][:, 0:n]


def z_score(dtr, dte=None):
    mu = mcol(dtr.mean(1))
    std = mcol(dtr.std(1))
    dtr = (dtr - mu) / std
    if dte is not None:
        dte = (dte - mu) / std
    return dtr, dte


def lda(d, l, n=1):
    mu = mcol(d.mean(axis=1))
    sb = np.zeros((d.shape[0], d.shape[0]))
    sw = np.zeros((d.shape[0], d.shape[0]))
    for c in range(l.max()+1):
        d_c = d[:, l == c]
        mu_c = mcol(d_c.mean(axis=1))
        sb += d_c.shape[1] * np.dot((mu_c - mu), (mu_c - mu).T)
        sw += np.dot((d_c - mu_c), (d_c - mu_c).T)
    sb /= d.shape[1]
    sw /= d.shape[1]
    _, u = scipy.linalg.eigh(sb, sw)
    w = u[:, ::-1][:, 0:n]
    return w


def load(file):
    d = []
    l = []
    with open(file, "r") as f:
        for line in f:
            attributes = line.split(",")[:-1:1]
            properties = mcol(np.array([float(i) for i in attributes]))
            d.append(properties)
            label = line.split(",")[-1].replace("\n", "")
            l.append(label)
    return np.hstack(d), np.array(l, dtype=np.int32)


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape([1, v.size])


def logpdf_GAU_ND_col(sample, mu, cov):
    xc = sample - mu        # center sample

    n_dim = sample.shape[0]     # number of dimensions
    const = n_dim * np.log(2 * np.pi)
    logdet = np.linalg.slogdet(cov)[1]      # determinant
    prec_mat = np.linalg.inv(cov)       # precision matrix
    v = np.dot(xc.T, np.dot(prec_mat, xc))
    return -0.5 * const - 0.5 * logdet - 0.5 * v


def logpdf_GAU_ND(d, mu, cov):
    log_densities = []
    for i in range(d.shape[1]):
        density = logpdf_GAU_ND_col(mcol(d[:, i]), mu, cov)
        log_densities.append(density)
    return np.hstack(log_densities)


def k_fold(d, l, k, model, p=None, seed=0, pca_m=None, zscore=False, calibration=False, fusion=False, model_desc=None):  # svm_params è c se lineare, parametri se kernel polinomiale, gamma se kernel rbf

    pca_desc = ""
    z_score_desc = ""

    n_test = math.ceil(d.shape[1]/k)

    k = math.ceil(d.shape[1] / n_test)

    np.random.seed(seed)
    idx = np.random.permutation(d.shape[1])

    rd = d[:, idx]  # reordered dataset
    rl = l[idx]  # reodered labels

    start = 0
    stop = n_test

    score = []

    for ki in range(k):
        # print(f"k_fold: Iterazione {ki + 1}")

        # DEFINIZIONE TEST SET
        i_test = range(start, stop, 1)
        dte = rd[:, i_test]
        lte = rl[i_test]

        # DEFINIZIONE TRAINING SET
        i_train = []
        for i in range(rd.shape[1]):
            if i not in i_test:
                i_train.append(i)
        dtr = rd[:, i_train]

        if zscore:
            dtr, dte = z_score(dtr, dte)
            z_score_desc = "_zscore"

        if pca_m is not None:
            # PCA
            p1 = pca(dtr, pca_m)
            dtr = np.dot(p1.T, dtr)
            dte = np.dot(p1.T, dte)
            pca_desc = f"_pca_{pca_m}"

        ltr = rl[i_train]

        model.set_data(dtr, ltr, dte, lte)
        model.train()
        score.append(model.get_scores())

        start += n_test
        stop += n_test

        if stop > d.shape[1]:
            stop = d.shape[1]

    score = np.hstack(score)
    preshuffle_score = np.zeros(score.shape)
    for i in range(d.shape[1]):
        preshuffle_score[idx[i]] = score[i]

    if not calibration:
        if not fusion:
            if not os.path.exists("score_models/" + model.folder()):
                os.makedirs("score_models/" + model.folder())
            np.save(f"score_models/{model.folder()}/{model.description()}{pca_desc}{z_score_desc}", preshuffle_score)
        else:
            if not os.path.exists("fusion_models"):
                os.makedirs("fusion_models")
            np.save(f"fusion_models/{model_desc}", preshuffle_score)
    else:
        preshuffle_score -= np.log(p/(1-p))
        if not os.path.exists("calibrated_score_models"):
            os.makedirs("calibrated_score_models")
        np.save(f"calibrated_score_models/{model_desc}", preshuffle_score)
    return preshuffle_score


def compute_min_dcf(scores, l, prior, cfn, cfp):
    thresholds = np.concatenate([np.array([-np.inf]), np.sort(scores), np.array([np.inf])])
    all_dcf = np.zeros(thresholds.shape)

    for i in range(thresholds.shape[0]):
        pl = predict_labels(scores, thresholds[i])
        conf_matrix = get_confusion_matrix(pl, l, l.max() + 1)

        all_dcf[i] = compute_dcf(conf_matrix, cfn, cfp, prior)

    return all_dcf.min()


def predict_labels(scores, th):
    labels = np.zeros(scores.shape[0])
    for i in range(scores.shape[0]):
        if scores[i] >= th:
            labels[i] += 1
    return np.array(labels, dtype=np.int32)


def get_confusion_matrix(pl, al, size):  # predicted and actual labels
    conf_matrix = np.zeros([size, size])

    for i in range(pl.shape[0]):
        conf_matrix[pl[i], al[i]] += 1

    return conf_matrix


def dcf_normalized(dcfu, pi, cfn, cfp):
    dcf1 = cfn * pi
    dcf2 = cfp * (1 - pi)

    if dcf1 < dcf2:
        return dcfu / dcf1
    return dcfu / dcf2


def compute_dcf(confusion_matrix, cfn, cfp, p):
    fnr = confusion_matrix[0, 1] / confusion_matrix[:, 1].sum()
    fpr = confusion_matrix[1, 0] / confusion_matrix[:, 0].sum()
    dcfu = p * cfn * fnr + (1 - p) * cfp * fpr
    return dcf_normalized(dcfu, p, cfn, cfp)
