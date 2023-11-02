from models.gmm.gmm_training_models import *
import numpy as np
from general.utils import k_fold, compute_min_dcf
import os

def cross_validation_for_gmm(d, l, model, zscore, pca):
    gmm_values = [range(6)]
    for gmm in gmm_values:
        model.set_values(2**gmm)
        print(f"g_num = {model.g_num} ")
        score = k_fold(d, l, 5, model, seed=27, zscore=zscore, pca_m=pca)

def compute_min_dcf_for_all_gmm(l, model, zscore, pca, name):  # name = model + k+ (gamma or constant or other)
    cfn = 1
    cfp = 1
    min_dcf_list = []
    prior = [0.5, 0.1, 0.9]
    gmm_values = []

    for p in prior:
        min_dcf = []
        for gmm in gmm_values:
            print(f"prior = {p}, gmm = {gmm}")
            model.set_values(c, pt)
            score = np.load(f"score_models/{model.folder()}/{model.description}{f"pca_{pca}" if pca else ""}{"_zscore" if zscore else ""}.npy")
            m_dcf = compute_min_dcf(score, l, p, cfn, cfp)
            min_dcf.append(m_dcf)
        min_dcf_list.append(min_dcf)
    if not os.path.exists("min_dcf_models/" + model.folder()):
        os.makedirs("min_dcf_models/" + model.folder())
    np.save(f"min_dcf_models/{model.folder()}/{name}{f"pca_{pca}" if pca else ""}{"_zscore" if zscore else ""}", min_dcf_list)

def cv_gmm_domain(d, l, zscore=False, pca=None):
    gmm = GMM(preprocess="z_score" if zscore else ""+"pca" if pca is not None else "")
    cross_validation_for_gmm(d, l, gmm, zscore, pca)

def cv_gmm_Tied(d, l, zscore=False, pca=None):
    gmm_tied = GMMTied(preprocess="z_score" if zscore else ""+"pca" if pca is not None else "")
    cross_validation_for_gmm(d, l, gmm_tied, zscore, pca)

def cv_gmm_Diag(d, l, zscore=False, pca=None):
    gmm_diag = GMMDiag(preprocess="z_score" if zscore else ""+"pca" if pca is not None else "")
    cross_validation_for_gmm(d, l, gmm_diag, zscore, pca)

def cv_gmm_TiedDiag(d, l, zscore=False, pca=None):
    gmm_tied_diag = GMMTiedDiag(preprocess="z_score" if zscore else ""+"pca" if pca is not None else "")
    cross_validation_for_gmm(d, l, gmm_tied_diag, zscore, pca)

def gmm_pca(d, l, model, zscore=False):
    for pca in [None, 10, 11, 12]:
        if pca is None:
            print("____without PCA_____")
        else:
            print(f"____PCA: {pca}_____")
            if zscore:
                model.set_preprocess("zscore/pca")
            else:
                model.set_preprocess("raw/pca")
        cross_validation_for_gmm(d, l, model, zscore, pca)
        model.__init__()

def cross_validation_for_all_gmm(d, l):
    for model in [GMM(), GMMTied(), GMMDiag(), GMMTiedDiag()]:
        print(f"_____{model.folder()} without Z Score____")
        gmm_pca(d, l, model)
        print(f"_____{model.folder()} with Z Score____")
        gmm_pca(d, l, model, zscore=True)

