import numpy as np
import matplotlib as plt
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


def scatter2D(D, L):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i != j:
                plt.figure()
                for c in range(L.max() + 1):
                    D_c = D[:, L == c]
                    x1 = D_c[i, :]
                    x2 = D_c[j, :]
                    plt.scatter(x1, x2, label=c)
                plt.xlabel(f'Dimension {i}')
                plt.xlabel(f'Dimension {j}')
                plt.legend()


def scatter3D(D, L):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            for k in range(D.shape[0]):
                if i != j and i != k and j != k:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for c in range(L.max() + 1):
                        D_c = D[:, L == c]
                        x = D_c[i, :]
                        y = D_c[j, :]
                        z = D_c[k, :]
                        ax.scatter(x, y, z, label=c)
                        ax.set_xlabel('X Label')
                        ax.set_ylabel('Y Label')
                        ax.set_zlabel('Z Label')
                    plt.legend()


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