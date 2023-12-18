from models.gmm.gmm_training_models import GMM, GMMTied, GMMDiag, GMMTiedDiag
from general.evaluation import get_evaluation_scores
import numpy as np


def evaluate_GMM(dtr, ltr, dte, lte, cfn, cfp):
    models_list = [(GMM(4), 11, False, "GMM"),   #(model, pca, zscore, score_folder)
                   (GMMTied(8), 11, True, "Tied_GMM_"),
                   (GMMDiag(4), 11, False, "Diag_GMM_"),
                   (GMMTiedDiag(16), 12, False, "TiedDiag_GMM_")]
    for model in models_list:
        score_path = f"score_models/{model[3]}/{"z_score" if model[2] else "raw"}{"/pca" if not model[2] and model[1] is not None else ""}/{model[0].description()}{f"_pca_{model[1]}" if model[1] is not None else ""}{"_zscore" if model[2] else ""}.npy"
        print(score_path)
        training_scores = np.load(score_path)
        test_scores = get_evaluation_scores(dtr, ltr, dte, lte, model[0], pca_m=model[1], zscore=model[2], model_desc=model[3])
