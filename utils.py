import numpy
import numpy as np
import math

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


def k_fold(d, l, k, model, p, cfn, cfp, seed=27):

    n_test = math.ceil(d.shape[1]/k)

    k = math.ceil(d.shape[1] / n_test)

    np.random.seed(seed)  # se eseguo il codice 2 volte il risultato non cambia
    idx = np.random.permutation(d.shape[1])

    rd = d[:, idx]  # reordered dataset
    rl = l[idx]  # reodered labels

    start = 0
    stop = n_test

    score = []

    for ki in range(k):
        # print(f"Iterazione {k}")

        # DEFINIZIONE TEST SET
        i_test = range(start, stop, 1)
        dte = rd[:, i_test]

        # DEFINIZIONE TRAINING SET
        i_train = []
        for i in range(rd.shape[1]):
            if i not in i_test:
                i_train.append(i)
        dtr = rd[:, i_train]
        ltr = rl[i_train]

        score.append(model(dtr, ltr, dte))

        start += n_test
        stop += n_test

        if stop > d.shape[1]:
            stop = d.shape[1]

    score = np.hstack(score)

    preshuffle_score = numpy.zeros(score.shape)
    for i in range(d.shape[1]):
        preshuffle_score[idx[i]] = score[i]

    ordered_score = np.sort(score)
    all_dcf = np.zeros(score.shape)

    for i in range(ordered_score.shape[0]):
        th = ordered_score[i]
        pl = predict_labels(preshuffle_score, th)
        conf_matrix = get_confusion_matrix(pl, l, l.max() + 1)

        all_dcf[i] = compute_dcf(conf_matrix, cfn, cfp, p)

    return all_dcf.min()

def predict_labels(scores, th):
    labels = np.zeros(scores.shape[0])
    for i in range(scores.shape[0]):
        if scores[i] > th:
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
