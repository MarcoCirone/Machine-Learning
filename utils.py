import numpy as np

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