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
                plt.text(j, i, f'{heatmap[i, j]: .1f}', ha='center', va='center',
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
    # gamma = [0.001, 0.01, 0.1]
    # k = [0.1, 1, 10]
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



def plot_min_dcfs_svm_for_evaluation(min_dcf_list1, min_dcf_list2, description, values):
    label = "C"
    e = [0.5, 0.1, 0.9]
    color = ["orange", "blue", "red"]
    # gamma = [0.001, 0.01, 0.1]

    for i in range(len(e)):
        plt.plot(values, min_dcf_list1[i], label=f"val -> p={e[i]}", linestyle='dotted', color=color[i])
        plt.plot(values, min_dcf_list2[i], label=f"eval -> p={e[i]}", color=color[i])
    plt.xscale("log")
    plt.xlabel(label)
    plt.ylabel("minDCF")
    plt.xlim([values[0], values[-1]])
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(f"figures/{description}")
    plt.close()

def plot_min_dcfs_gmm(min_dcf_list, min_dcf_list_zscore, description, values):

    folder = "GMM"
    plt.figure()
    plt.xlabel("Components")
    plt.ylabel("minDCF")

    plt.bar(np.arange(values), height=min_dcf_list, width=0.3, label="Raw", color="Blue")
    plt.bar(np.arange(values)+0.3, height=min_dcf_list_zscore, width=0.3, label="Zscore", color="Red")
    plt.xticks([i + 0.15 for i in range(values)], [2**i for i in np.array(range(1, values+1))])
    plt.legend()
    if not os.path.exists("figures/GMM"):
        os.makedirs("figures/GMM")
    plt.savefig(f"figures/{folder}/{description}")
    plt.close()

def plot_min_dcfs_gmm_for_evaluation(min_dcf_list, min_dcf_list_zscore, min_dcf_list2, min_dcf_list_zscore2, description, values):
    plt.figure()
    plt.xlabel("Components")
    plt.ylabel("minDCF")

    plt.bar(np.arange(values)*1.5, height=min_dcf_list2, width=0.3, label="Val -> Raw", color="Blue", alpha=0.5)
    plt.bar(np.arange(values)*1.5+0.3, height=min_dcf_list_zscore2, width=0.3, label="Val -> Zscore", color="Red", alpha=0.5)
    plt.bar(np.arange(values)*1.5+0.6, height=min_dcf_list, width=0.3, label="Eval -> Raw", color="Blue")
    plt.bar(np.arange(values)*1.5+0.9, height=min_dcf_list_zscore, width=0.3, label="Eval -> Zscore", color="Red")
    plt.xticks([i*1.5 + 0.45 for i in range(values)], [2**i for i in np.array(range(1, values+1))])
    plt.legend()
    plt.savefig(f"figures/{description}")
    plt.close()

def plot_det_roc(scores, l, labels, train=False):
    fpr_list = []
    fnr_list = []
    tpr_list = []
    print("starting det/roc computation")
    for s in scores:
        thresholds = np.concatenate([np.array([-np.inf]), np.sort(s), np.array([np.inf])])
        fpr_model = []
        fnr_model = []
        tpr_model = []
        for th in thresholds:
            pl = predict_labels(s, th)
            conf_matrix = get_confusion_matrix(pl, l, l.max() + 1)
            fpr, fnr = conf_matrix[1, 0] / conf_matrix[:, 0].sum(), conf_matrix[0, 1] / conf_matrix[:, 1].sum()
            tpr = 1 - fnr
            fpr_model.append(fpr)
            fnr_model.append(fnr)
            tpr_model.append(tpr)
        fpr_list.append(fpr_model)
        fnr_list.append(fnr_model)
        tpr_list.append(tpr_model)
    print("finish det/roc computation")
    # plot det
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True)
    for i, (fpr, fnr) in enumerate(zip(fpr_list, fnr_list)):
        plt.plot(fpr, fnr, label=labels[i])
    plt.legend()
    if train:
        if not os.path.exists("figures"):
            os.makedirs("figures")
        plt.savefig(f"figures/det_plots")
    else:
        if not os.path.exists("figures/evaluation"):
            os.makedirs("figures/evaluation")
        plt.savefig(f"figures/evaluation/det_plots")

    # plot roc
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xscale('log')
    plt.yscale('log')


    plt.grid(True)
    for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
        plt.plot(fpr, tpr, label=labels[i])
    plt.legend()
    if train:
        if not os.path.exists("figures"):
            os.makedirs("figures")
        plt.savefig(f"figures/roc_plots")
    else:
        if not os.path.exists("figures/evaluation"):
            os.makedirs("figures/evaluation")
        plt.savefig(f"figures/evaluation/roc_plots")

def plot_bayes_error(scores, ltr, cfn, cfp, model_desc, train=True):
    eff_prior_log_odds = np.linspace(-4, 4, 31)
    k = 0
    dcf = []
    mindcf = []
    for e in effPriorLogOdds:
        # print(k)
        k += 1
        # CALCOLO DCF EFFETTIVO
        pi = 1 / (1 + np.exp(-e))
        th = -np.log(pi / (1 - pi))
        pl = predict_labels(scores, th)
        conf_matrix = get_confusion_matrix(pl, ltr, 2)
        dcf.append(compute_dcf(conf_matrix, cfn, cfp, pi))
        mindcf.append(compute_min_dcf(scores, ltr, pi, cfn, cfp))

    plt.figure()
    plt.plot(eff_prior_log_odds, dcf, label="actDCF", color="r")
    plt.plot(eff_prior_log_odds, mindcf, label="minDCF", color="b")
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.xlabel("Threshold")
    plt.ylabel("DCF")
    plt.legend()
    if train:
        if not os.path.exists("figures/bayes_error_plots"):
            os.makedirs("figures/bayes_error_plots")
        plt.savefig(f"figures/bayes_error_plots/{model_desc}")
    else:
        if not os.path.exists("figures/evaluation/bayes_error_plots"):
            os.makedirs("figures/evaluation/bayes_error_plots")
        plt.savefig(f"figures/evaluation/bayes_error_plots/{model_desc}")
    plt.show()
