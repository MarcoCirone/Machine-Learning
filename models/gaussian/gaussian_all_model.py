from models.gaussian.gaussians_training_models import *
import numpy as np
from general.utils import k_fold

def compute_gaussian(d, l, model, zscore, pca):
    cfn = 1
    cfp = 1

    prior = [0.5, 0.1, 0.9]

    for p in prior:
        min_dcf = k_fold(d, l, 5, model, p, cfn, cfp, seed=27, zscore=zscore, pca_m=pca)
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
