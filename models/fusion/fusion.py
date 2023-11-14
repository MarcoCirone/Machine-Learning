import numpy

from models.logistic_regression.log_reg_training_models import LR
from general.utils import k_fold


def fusion(scores_list, l, prior, model_desc):
    d = numpy.vstack(scores_list)
    return k_fold(d, l, 5, LR(0, prior), p=prior, seed=27, model_desc=model_desc)