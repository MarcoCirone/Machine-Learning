import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("./")
from general.utils import k_fold

def hist(d, l):
    for i in range(d.shape[0]):
        plt.figure()
        for c in range(l.max() + 1):
            x = d[i, l == c]
            plt.hist(x.reshape(x.size, ), alpha=0.4, label="c")
        plt.xlabel(f'Dimension {i + 1}')
        plt.legend()
        plt.show()


def scatter_2d(d, l):
    for i in range(d.shape[0]):
        for j in range(d.shape[0]):
            if i != j:
                plt.figure()
                for c in range(l.max() + 1):
                    d_c = d[:, l == c]
                    x1 = d_c[i, :]
                    x2 = d_c[j, :]
                    plt.scatter(x1, x2, label=c)
                plt.xlabel(f'Dimension {i}')
                plt.xlabel(f'Dimension {j}')
                plt.legend()


def scatter_3d(d, l):
    for i in range(d.shape[0]):
        for j in range(d.shape[0]):
            for k in range(d.shape[0]):
                if i != j and i != k and j != k:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for c in range(l.max() + 1):
                        d_c = d[:, l == c]
                        x = d_c[i, :]
                        y = d_c[j, :]
                        z = d_c[k, :]
                        ax.scatter(x, y, z, label=c)
                        ax.set_xlabel('X Label')
                        ax.set_ylabel('Y Label')
                        ax.set_zlabel('Z Label')
                    plt.legend()
def plot_min_dcfs(dtr, ltr, cfn, cfp, model, pt=None, svm_params=None, reg_term=None, pca_m=None, seed=0):
    values = np.logspace(-5, 5, num=31)
    label = "C"
    min_dcf_list = []
    e = [0.5, 0.1, 0.9]
    k = 0

    if svm_params is None:
        values = np.logspace(-5, 5, num=51)
        label = "$lambda$"

    for eff_p in e:
        mins = []
        for v in values:

            if svm_params is None:
                reg_term = v
            else:
                svm_params[1] = v

            dcf_min = k_fold(dtr, ltr, 5, model, eff_p, cfn, cfp, seed=seed, pt=pt, reg_term=reg_term, svm_params=svm_params, pca_m=pca_m)
            mins.append(dcf_min)
            k += 1
            print(f"Iterazione {k}: prior= {eff_p} {label}= {v} => min_dcf= {dcf_min}")
        min_dcf_list.append(mins.copy())
    for i in range(len(min_dcf_list)):
        plt.plot(values, min_dcf_list[i], label=f"eff_p={e[i]}")
    plt.xscale("log")
    plt.xlabel(label)
    plt.ylabel("minDCF")
    plt.xlim([c_values[0], c_values[-1]])
    plt.legend()
