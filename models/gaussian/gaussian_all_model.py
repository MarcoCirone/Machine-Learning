from models.gaussian.gaussians_training_models import *
import numpy as np
from general.utils import k_fold, compute_min_dcf

def compute_gaussian(d, l, model, zscore, pca):
    prior = [0.5, 0.1, 0.9]
    cfn = 1
    cfp = 1

    for p in prior:
        scores = k_fold(d, l, 5, model, p, seed=27, zscore=zscore, pca_m=pca)
        min_dcf = compute_min_dcf(scores, l, p, cfn, cfp)
        print(f"prior = {p}, min_dcf = {min_dcf}")

def gaussians_pca(d, l, model, zscore=False):
    for pca in [None, 10, 11, 12]:
        if pca is None:
            print("____without PCA_____")
        else:
            print(f"____PCA: {pca}_____")
        compute_gaussian(d, l, model, zscore, pca)
        model.__init__()

def gaussians(d, l):
    for model in [MVG(), MVGTied(), NaiveBayes(), NBTied()]:
        print(f"_____{model.folder()} without Z Score____")
        gaussians_pca(d, l, model)
        print(f"_____{model.folder()} with Z Score____")
        gaussians_pca(d, l, model, zscore=True)
