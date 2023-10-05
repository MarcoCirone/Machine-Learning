import numpy as np
import math

def load(file):
    D = []
    L = []
    with open(file, "r") as f:
        for line in f:
            attributes = line.split(",")[:-1:1]
            properties = mcol(np.array([float (i) for i in attributes]))
            D.append(properties)
            label = line.split(",")[-1].replace("\n", "")
            L.append(label)
    return np.hstack(D), np.array(L, dtype = np.int32)

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape([1, v.size])




def logpdf_GAU_ND_col(sample, mu, cov):
    xc = sample - mu #center sample

    n_dim = sample.shape[0] #number of dimensions
    const = n_dim * np.log(2 * np.pi)
    logdet = np.linalg.slogdet(cov)[1]  # determinante
    prec_mat = np.linalg.inv(cov) #precision matrix
    v = np.dot(xc.T, np.dot(prec_mat, xc))
    return -0.5 * const - 0.5 * logdet - 0.5 * v


def logpdf_GAU_ND(d, mu, cov):
    log_densities = []
    for i in range(d.shape[1]):
        density = logpdf_GAU_ND_col(mcol(d[:, i]), mu, cov)
        log_densities.append(density)
    return np.hstack(log_densities)


def KFold(D, L, K, model, p, Cfn, Cfp, seed=27):
    t=-np.log(p*Cfn/((1-p)*Cfp))

    nTest = math.ceil(D.shape[1] / K)

    K = math.ceil(D.shape[1] / nTest)

    np.random.seed(seed)  # se eseguo il codice 2 volte il risultato non cambia
    idx = np.random.permutation(D.shape[1])

    RD = D[:, idx]  # reordered dataset
    RL = L[idx]  # reodered labels

    start = 0
    stop = nTest

    S = []

    for k in range(K):
        # print(f"Iterazione {k}")

        # DEFINIZIONE TEST SET
        iTest = range(start, stop, 1)
        DTE = RD[:, iTest]
        LTE = RL[iTest]

        # DEFINIZIONE TRAINING SET
        iTrain = []
        for i in range(RD.shape[1]):
            if i not in iTest:
                iTrain.append(i)
        DTR = RD[:, iTrain]
        LTR = RL[iTrain]

        S.append(model(DTR, LTR, DTE))

        start += nTest
        stop += nTest

        if stop > D.shape[1]:
            stop = D.shape[1]

    S = np.hstack(S)

    So = np.zeros(S.shape)

    for i in range(D.shape[1]):
        So[:, idx[i]] = S[:, i]

    p_labels=predict_labels(np.array(So, dtype=np.float32), t)

    conf_matrix=getConfusionMatrix(p_labels, L, L.max()+1)

    return conf_matrix

def predict_labels(scores, th):
    labels=np.zeros(scores.shape[0])
    for i in range(scores.shape[0]):
        if scores[i]>th:
            labels[i]+=1
    return np.array(labels, dtype=np.int32)


def getConfusionMatrix(PL, AL, size):  # predicted and actual labels
    conf_matrix = np.zeros([size, size])

    for i in range(PL.shape[0]):
        conf_matrix[PL[i], AL[i]] += 1

    return conf_matrix