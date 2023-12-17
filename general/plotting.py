import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import os
from general.utils import *

sys.path.append("./")
from general.utils import k_fold, pca


def hist(d, l, labels):
    for i in range(d.shape[0]):
        plt.figure()
        for c in range(2):
            x = d[i, l == c]
            plt.hist(x.reshape(x.size, ), alpha=0.4, label=labels[c], bins=70, density=True, linewidth=1.0)
        plt.xlabel(f'Dimension {i + 1}')
        plt.legend()
        plt.savefig("figures/lda_histograms_" + str(i))
        plt.close()


def plot_pca(d):
    s = pca(d, d.shape[0] + 1, eigen_values=True)
    sort_s = s[::-1]
    var = np.sum(sort_s)
    evr = np.cumsum(sort_s / var)
    plt.plot(range(1, s.size + 1), evr, marker='o')
    plt.xlabel('Dimensions')
    plt.ylabel('Fraction of explained variance')
    plt.grid(True)
    plt.savefig("figures/pca_var.png")
    plt.close()


def plot_heatmaps(d, l, labels):
    classes = [0, 1, None]
    colors = ["Blues", "Reds", "Greys"]
    for c in range(3):
        if classes[c] is None:
            data = d
        else:
            data = d[:, l == classes[c]]
        heatmap = np.zeros((d.shape[0], d.shape[0]))

        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                heatmap[i, j] = abs(scipy.stats.pearsonr(data[i, :], data[j, :])[0])
                heatmap[j, i] = heatmap[i, j]
        plt.figure()
        plt.xticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
        plt.yticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
        plt.imshow(heatmap, cmap=colors[c])
        rows, cols = heatmap.shape
        for i in range(rows):
            for j in range(cols):
                plt.text(j, i, f'{heatmap[i, j]:.1f}', ha='center', va='center',
                         color='white' if round(heatmap[i, j], 1) >= 0.8 else 'black')
        plt.colorbar()
        plt.savefig("figures/heatmaps/heatmaps_" + labels[c])
        plt.close()


def scatter_2d(d, l, labels):
    for i in range(d.shape[0]):
        for j in range(d.shape[0]):
            if i != j:
                plt.figure()
                for c in range(2):
                    d_c = d[:, l == c]
                    x1 = d_c[i, :]
                    x2 = d_c[j, :]
                    plt.scatter(x1, x2, label=labels[c])
                plt.xlabel(f'Dimension {i}')
                plt.ylabel(f'Dimension {j}')
                plt.legend()
                plt.savefig("figures/scatter_plots/scatter_plot_" + str(i)+"_"+str(j))
                plt.close()

def plot_min_dcfs_svm(min_dcf_list, description, values, pt=None):
    label = "C"
    folder = "SVM"
    e = [0.5, 0.1, 0.9]
    # e = [0.001, 0.01, 0.1]
    # e = [0.1, 1, 10]
    for i in range(len(min_dcf_list)):
        plt.plot(values, min_dcf_list[i], label=f"prior={e[i]}")
    plt.xscale("log")
    plt.xlabel(label)
    plt.ylabel("minDCF")
    plt.xlim([values[0], values[-1]])
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(f"figures/{folder}/{description}_pt_{pt}")
    plt.close()

def plot_min_dcfs_gmm(min_dcf_list, min_dcf_list_zscore, description, values):

    folder = "GMM"
    plt.figure()
    plt.xlabel("Components")
    plt.ylabel("minDCF")

    plt.bar(np.arange(values), height=min_dcf_list, width=0.3, label="Raw", color="Blue")
    plt.bar(np.arange(values)+0.3, height=min_dcf_list_zscore, width=0.3, label="Zscore", color="Red")
    plt.xticks([i + 0.15 for i in range(values)],[2**i for i in np.array(range(1,values+1))])
    plt.legend()
    if not os.path.exists("figures/GMM"):
        os.makedirs("figures/GMM")
    plt.savefig(f"figures/{folder}/{description}")
    plt.close()


def plot_bayes_error(scores, ltr, cfn, cfp, model_desc):
    effPriorLogOdds = np.linspace(-4, 4, 31)
    k = 0
    dcf = []
    mindcf = []
    for e in effPriorLogOdds:
        # print(k)
        k += 1
        # CALCOLO DCF EFFETTIVO
        pi = 1 / (1 + np.exp(-e))
        th = -np.log(pi / (1 - pi))
        PL = predict_labels(scores, th)
        conf_matrix = get_confusion_matrix(PL, ltr, 2)
        dcf.append(compute_dcf(conf_matrix, cfn, cfp, pi))
        mindcf.append(compute_min_dcf(scores, ltr, pi, cfn, cfp))

    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label="actDCF", color="r")
    plt.plot(effPriorLogOdds, mindcf, label="minDCF", color="b")
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.xlabel("Threshold")
    plt.ylabel("DCF")
    plt.legend()
    if not os.path.exists("figures/bayes_error_plots"):
        os.makedirs("figures/bayes_error_plots")
    plt.savefig(f"figures/bayes_error_plots/{model_desc}")
    plt.show()

