from models.gmm.gmm_training_models import *
import numpy as np
from general.plotting import plot_min_dcfs_gmm_for_evaluation
from general.utils import compute_min_dcf
from general.evaluation import get_evaluation_scores
import os

def evaluation_for_gmm(d, l, model, zscore, pca, dte=None, lte=None):
    gmm_values = list(range(1, 7))
    for gmm in gmm_values:
        model.set_values(2**gmm)
        print(f"g_num = {model.g_num} ")
        _score = get_evaluation_scores(d, l, dte, lte, model, zscore=zscore, pca_m=pca, model_desc=f"{model.description()}{"_zscore" if zscore else ""}")
def compute_min_dcf_for_all_gmm(l, model, zscore, pca, name):  # name = model + k+ (gamma or constant or other)

    cfn = 1
    cfp = 1
    min_dcf_list = []
    prior = [0.5]
    gmm_values = list(range(1, 7))

    for p in prior:
        min_dcf = []
        for gmm in gmm_values:
            print(f"prior = {p}, gmm = {2**gmm}")
            model.set_values(2**gmm)
            score = np.load(f"evaluation/scores/{model.description()}{"_zscore" if zscore else ""}.npy")
            m_dcf = compute_min_dcf(score, l, p, cfn, cfp)
            min_dcf.append(m_dcf)
        min_dcf_list.append(min_dcf)
    if not os.path.exists("evaluation/min_dicf"):
        os.makedirs("evaluation/min_dicf")
    np.save(f"evaluation/min_dicf/{name}{f"pca_{pca}" if pca else ""}{"_zscore" if zscore else ""}", min_dcf_list)

def plot_all_gmm(name, pca):

    min_dcf1 = np.load(f"evaluation/min_dicf/{name}{f"pca_{pca}" if pca else ""}.npy")
    min_dcf2 = np.load(f"min_dcf_models/{name}/{name}{f"pca_{pca}" if pca else ""}.npy")
    min_dcf1_zscore = np.load(f"evaluation/min_dicf/{name}{f"pca_{pca}" if pca else ""}_zscore.npy")
    min_dcf2_zscore = np.load(f"min_dcf_models/{name}/{name}{f"pca_{pca}" if pca else ""}_zscore.npy")

    plot_min_dcfs_gmm_for_evaluation(min_dcf1[0], min_dcf1_zscore[0], min_dcf2[0], min_dcf2_zscore[0], f"evaluation{name}", 6)


def gmm_model(d, l, model,  dte=None, lte=None):
    pca = 11
    for zscore in [False, True]:
        if zscore:
            print(f"_____{model.folder()} with Z Score____")
        else:
            print(f"_____{model.folder()} without Z Score____")
        print(f"____PCA: {pca}_____")

        model.set_preprocess("z_score/" if zscore else "raw/"+"pca" if pca is not None else "raw")
        evaluation_for_gmm(d, l, model, zscore=zscore, pca=pca, dte=dte, lte=lte)
        compute_min_dcf_for_all_gmm(lte, model, zscore, pca, model.description())
        model.__init__()
    plot_all_gmm(model.description(), pca)


def evaluation_for_all_gmm(d, l, dte=None, lte=None):
    for model in [GMM(), GMMTied()]:
        gmm_model(d, l, model, dte=dte, lte=lte)
