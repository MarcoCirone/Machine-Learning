from models.gmm.gmm_training_models import *
import numpy as np
from general.plotting import plot_min_dcfs_gmm
from general.utils import k_fold, compute_min_dcf
import os

def cross_validation_for_gmm(d, l, model, zscore, pca):
    gmm_values = list(range(1, 7))
    for gmm in gmm_values:
        model.set_values(2**gmm)
        print(f"g_num = {model.g_num} ")
        _score = k_fold(d, l, 5, model, seed=27, zscore=zscore, pca_m=pca)

def compute_min_dcf_for_all_gmm(l, model, zscore, pca, name):  # name = model + k+ (gamma or constant or other)
    cfn = 1
    cfp = 1
    min_dcf_list = []
    prior = [0.1, 0.5, 0.9]
    gmm_values = list(range(1, 7))

    for p in prior:
        min_dcf = []
        for gmm in gmm_values:
            print(f"prior = {p}, gmm = {2**gmm}")
            model.set_values(2**gmm)
            score = np.load(f"score_models/{model.folder()}/{model.description()}{f"_pca_{pca}" if pca else ""}{"_zscore" if zscore else ""}.npy")
            m_dcf = compute_min_dcf(score, l, p, cfn, cfp)
            min_dcf.append(m_dcf)
        min_dcf_list.append(min_dcf)
    if not os.path.exists(f"min_dcf_models/{name}"):
        os.makedirs("min_dcf_models/" + name)
    np.save(f"min_dcf_models/{name}/{name}{f"pca_{pca}" if pca else ""}{"_zscore" if zscore else ""}", min_dcf_list)

def plot_all_gmm(name, pca):

    min_dcf = np.load(f"min_dcf_models/{name}/{name}{f"pca_{pca}" if pca else ""}.npy")
    min_dcf_zscore = np.load(f"min_dcf_models/{name}/{name}{f"pca_{pca}" if pca else ""}_zscore.npy")
    plot_min_dcfs_gmm(min_dcf[0], min_dcf_zscore[0], f"{name}_pca_{pca}", 6)


def gmm_pca(d, l, model, zscore=False):
    for pca in [None, 10, 11, 12]:
        if pca is None:
            print("____without PCA_____")
        else:
            print(f"____PCA: {pca}_____")

        model.set_preprocess("z_score/" if zscore else "raw/"+"pca" if pca is not None else "raw")
        cross_validation_for_gmm(d, l, model, zscore, pca)
        compute_min_dcf_for_all_gmm(l, model, zscore, pca, model.description())
        plot_all_gmm(model.description(), pca)
        model.__init__()

def cross_validation_for_all_gmm(d, l):
    for model in [GMM(), GMMTied(), GMMTiedDiag(), GMMDiag()]:
        print(f"_____{model.folder()} without Z Score____")
        gmm_pca(d, l, model)
        print(f"_____{model.folder()} with Z Score____")
        gmm_pca(d, l, model, zscore=True)
